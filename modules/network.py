"""
Log:
9.16:
 GT: 45+ 3 + 3; [R+T] * [3 poses+T]--> poses + T1
pred: 6--> 3*3 for loss, 3*3--> euler for prediction;
3 trans = average;

"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from manopth.manolayer import ManoLayer

import __init__ as booger
from global_info import global_info
from modules.segmentator import ModelBuilder
from common.train_utils import AverageMeter, parse_devices, warmupLR, Logger, save_to_log, make_log_img, decide_checkpoints, save_batch_nn
from models.loss_helper import compute_vote_loss, compute_objectness_loss, compute_box_and_sem_cls_loss
from models.loss_helper import get_loss
from modules.ioueval import *
from modules.network_base import NetworkBase

infos           = global_info()
my_dir          = infos.base_path
grasps_meta     = infos.grasps_meta
mano_path       = infos.mano_path

def breakpoint(id=0):
    import pdb;pdb.set_trace()

class Network(NetworkBase):
    def __init__(self, gpu, cfg): # with seperate gpu for distributed training
        NetworkBase.__init__(self, gpu, cfg)

    def train_epoch(self, gpu_id, loader, model, epoch, cfg, modules=None):
        """
        if we have pretrained features, we could use modules
        """
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        losses_meter = {}
        for key in self.total_loss_dict.keys():
            losses_meter[key] = AverageMeter()

        model.train(not cfg.TRAIN.fix_bn)
        torch.cuda.empty_cache() # release memory to avoid OOM error

        tic = time.time()

        for i, batch in enumerate(loader):
            if cfg.is_debug and i > 25:
                print('good with train_epoch!!')
                break
            self.fill_gpu_batch_data(gpu_id, batch)

            # prepare for training
            model.zero_grad()
            pred_dict = model(batch['P'])

            if cfg.pred_mano:
                with torch.no_grad():
                    handvertices, handjoints = model.mano_layer_gt(th_pose_coeffs=batch['regression_params'])
                    batch['handvertices']     = handvertices/200
                    batch['handjoints']       = handjoints/200
                    batch['regressionT']      = torch.mean(batch['hand_joints'] - batch['handjoints'], dim=1)
            # Compute loss and gradients, update parameters.
            loss_dict = self.compute_loss(pred_dict, batch, cfg)

            # combine different losses
            loss_total= self.collect_losses(loss_dict)

            # >>>>>>>>>>>  backward
            loss_total.backward()
            for _, optimizers in self.optimizers_dict.items():
                for optimizer in optimizers:
                    if optimizer:
                        optimizer.step()

            # get iteration steps to adjust learning rate,
            cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
            if self.schedulers_dict is None:
                scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
                cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
                cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr
                for _, optimizers in self.optimizers_dict.items():
                    self.adjust_learning_rate(optimizers, cfg) # adjust according to step
            else:
                schedulers = self.schedulers_dict[list(self.schedulers_dict.keys())[0]]
                for j, scheduler in enumerate(schedulers):
                    if scheduler is not None:
                        scheduler.step()
                for _, optimizers in self.optimizers_dict.items():
                    if optimizers[0] is not None:
                        for g in optimizers[0].param_groups:
                            cfg.TRAIN.running_lr_encoder = g["lr"]
                    if optimizers[1] is not None:
                        for g in optimizers[1].param_groups:
                            cfg.TRAIN.running_lr_decoder = g["lr"]

            # >>>>>>>>>>>>>>>> for logging use only
            for key, loss_item in self.total_loss_dict.items():
                if isinstance(loss_item, int):
                    continue
                losses_meter[key].update(loss_item.item(), 1)
            batch_time.update(time.time() - tic)
            msg = 'Loss: {loss:.3f} m:{miou_loss:.3f} r:{regression_loss:.3f}, R:{regressionR_loss:.3f}, T:{regressionT_loss:.3f}, hv: {handvertices_loss:.3f}, hj: {handjoints_loss:.3f}'.format(
                                      loss=loss_total, \
                                      miou_loss=self.total_loss_dict['partcls_loss'], \
                                      regression_loss=self.total_loss_dict['regression_loss'],\
                                      regressionR_loss=self.total_loss_dict['regressionR_loss'],\
                                      regressionT_loss=self.total_loss_dict['regressionT_loss'],\
                                      handvertices_loss=self.total_loss_dict['handvertices_loss'],\
                                      handjoints_loss=self.total_loss_dict['handjoints_loss'])

            print('Training: [{0}][{1}/{2}] | ''Lr: {lr_encoder:.3e} {lr_decoder:.3e} | '.format(epoch, i, len(loader), \
                      lr_encoder=cfg.TRAIN.running_lr_encoder, \
                      lr_decoder=cfg.TRAIN.running_lr_decoder), msg)

        print(f'Epoch {epoch} has : ')
        for key, loss_item in losses_meter.items():
            self.infos[key] = loss_item.avg
            if cfg.verbose:
                print('---training ', key, self.infos[key])
        self.infos['learning_rate'] = cfg.TRAIN.running_lr_decoder
        save_to_log(logdir=cfg.log_dir + '/tb/train', logger=self.tb_logger['train'], info=self.infos, epoch=epoch, img_summary=cfg.TRAIN.save_scans)
        print('')

        return 1-self.total_loss_dict['partcls_loss'], loss_total

    def valid_epoch(self, gpu_id, loader, model, epoch, cfg, modules=None, prefix='valid', save_pred=False):
        """
        if we have pretrained features, we could use modules
        """
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        losses_meter = {}
        for key in self.total_loss_dict.keys():
            losses_meter[key] = AverageMeter()
        model.eval()
        model.zero_grad()
        torch.cuda.empty_cache() # release memory to avoid OOM error
        if epoch > 1:
            cfg.TRAIN.confidence_loss_multiplier = 1.0
        tic = time.time()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if cfg.is_debug and i > 25:
                    print('good with valid_epoch!!')
                    break
                self.fill_gpu_batch_data(gpu_id, batch)

                pred_dict = model(batch['P'])
                if cfg.pred_mano:
                    handvertices, handjoints = model.mano_layer_gt(th_pose_coeffs=batch['regression_params'])
                    batch['handvertices']     = handvertices/200
                    batch['handjoints']       = handjoints/200
                    batch['regressionT']      = torch.mean(batch['hand_joints'] - batch['handjoints'], dim=1)
                loss_dict = self.compute_loss(pred_dict, batch, cfg)

                if cfg.pred_mano:
                    handvertices, handjoints = model.mano_layer_gt(th_pose_coeffs=batch['regression_params'])
                    batch['handvertices']     = handvertices/200
                    batch['handjoints']       = handjoints/200

                # # combine different losses
                loss_total= self.collect_losses(loss_dict)
                loss_total = loss_total.mean()

                for key, loss_item in self.total_loss_dict.items():
                    if isinstance(loss_item, int):
                        continue
                    losses_meter[key].update(loss_item.item(), 1)
                batch_time.update(time.time() - tic)

                # >>>>>>>>>>>>> logging
                msg = 'Loss: {loss:.3f} m:{miou_loss:.3f} r:{regression_loss:.3f}, R:{regressionR_loss:.3f}, T:{regressionT_loss:.3f}, hv: {handvertices_loss:.3f}, hj: {handjoints_loss:.3f}'.format(
                                      loss=loss_total, \
                                      miou_loss=self.total_loss_dict['partcls_loss'], \
                                      regression_loss=self.total_loss_dict['regression_loss'],\
                                      regressionR_loss=self.total_loss_dict['regressionR_loss'],\
                                      regressionT_loss=self.total_loss_dict['regressionT_loss'],\
                                      handvertices_loss=self.total_loss_dict['handvertices_loss'],\
                                      handjoints_loss=self.total_loss_dict['handjoints_loss'])
                print('---Val: [{0}][{1}/{2}] | '.format(epoch, i, len(loader)), msg)

                # >>>>>>>>>>>>> save
                if save_pred:
                    save_batch_nn(cfg.name_model, pred_dict, batch, loader.dataset.basename_list, save_dir=cfg.log_dir + f'/preds/{prefix}/')

        for key, loss_item in losses_meter.items():
            self.infos[key] = loss_item.avg
            if cfg.verbose:
                print(f'----{prefix} ', key, self.infos[key])
        save_to_log(logdir=cfg.log_dir + '/tb/' +prefix, logger=self.tb_logger[prefix], info=self.infos, epoch=epoch, img_summary=cfg.TRAIN.save_scans)
        if cfg.verbose:
            print('saving log to ', cfg.log_dir + '/' +prefix)

        print('')
        return 1 - self.total_loss_dict['partcls_loss'], losses_meter['total_loss'].avg
