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
from modules.mit_segmentator import ModelBuilder
from common.train_utils import AverageMeter, parse_devices, warmupLR, Logger, save_to_log, make_log_img, decide_checkpoints, save_batch_nn
from modules.losses import FocalLoss, compute_miou_loss, compute_nocs_loss, compute_vect_loss, compute_multi_offsets_loss, loss_geodesic
from models.loss_helper import compute_vote_loss, compute_objectness_loss, compute_box_and_sem_cls_loss
from models.loss_helper import get_loss
from modules.ioueval import *
from common.d3_utils import compute_rotation_matrix_from_euler, compute_euler_angles_from_rotation_matrices, compute_rotation_matrix_from_ortho6d
from utils.nn_distance import huber_loss

infos           = global_info()
my_dir          = infos.base_path
grasps_meta     = infos.grasps_meta
mano_path       = infos.mano_path

def breakpoint(id=0):
    import pdb;pdb.set_trace()

class EncoderDecoder(nn.Module):
    def __init__(self, net_enc, net_dec, head=None, head_names=None, cfg=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = net_enc
        self.decoder  = net_dec
        self.head     = head
        self.head_names= head_names
        self.pred_mano = cfg.pred_mano
        self.mano_layer= None
        if cfg.pred_mano:
            ncomps = 45
            self.mano_layer    = ManoLayer(mano_root=mano_path, root_rot_mode='ratation_6d', side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
            self.mano_layer_gt = ManoLayer(mano_root=mano_path, root_rot_mode='ratation_6d', side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    def forward(self, xyz1=None, inputs=None, mask=None, index=0):
        if inputs is not None:
            end_points = self.decoder(inputs)
            return end_points

        if self.backbone is None:
            net, bottle_neck = self.decoder(xyz1)
        else:
            net, bottle_neck = self.decoder(self.backbone(xyz1))
        pred_dict = {}

        for i, sub_head in enumerate(self.head):
            if 'regression' in self.head_names[i]:
                if bottle_neck.size(-1) !=1:
                    bottle_neck = bottle_neck.max(-1)[0]
                pred_dict[self.head_names[i]] = self.head[i](bottle_neck.view(-1, 1024))
            else:
                pred_dict[self.head_names[i]] = self.head[i](net)

        if self.pred_mano:
            regression_params = torch.split(pred_dict['regression_params'], [6, 45, 3], dim=1)
            pred_dict['regressionR'] = compute_rotation_matrix_from_ortho6d(regression_params[0])
            pred_dict['regressionT'] = regression_params[2]
            pred_dict['regression_params'] = regression_params[1] # focus on poses

            pose_coeffs = torch.cat((regression_params[0], regression_params[1]), dim=1)
            handvertices, handjoints = self.mano_layer.forward(th_pose_coeffs=pose_coeffs) #th_trans=pred_dict['regressionT']
            pred_dict['handvertices'] = handvertices/200
            pred_dict['handjoints']   = handjoints/200

        return pred_dict

class NetworkBase():
    def __init__(self, gpu, cfg): # with seperate gpu for distributed training
        self.rank = gpu
        self.config  = cfg
        self.use_gpu = True
        self.is_mixed  = False
        self.infos   = {} # for logging
        self.pred_joint= cfg.pred_joint
        self.pred_joint_ind = cfg.pred_joint_ind
        self.pred_hand = cfg.pred_hand
        self.tb_logger = {'train': Logger(cfg.log_dir + "/tb/train"), 'seen': Logger(cfg.log_dir + '/tb/seen'), 'unseen': Logger(cfg.log_dir + '/tb/unseen')}
        # >>>>>>>>>>>>>>> decide whether to use multi-gpu
        if cfg.num_gpus > 0:
            torch.cuda.set_device(gpu)
            if cfg.distributed:
                try:                                    # set up GPU
                    dist.init_process_group(backend='nccl', init_method='env://', world_size=cfg.TRAIN.world_size, rank=self.rank)
                except:
                    os.environ['MASTER_PORT'] = '8889'
                    dist.init_process_group(backend='nccl', init_method='env://', world_size=cfg.TRAIN.world_size, rank=self.rank)
            print('using gpu id: {} with rank: {}'.format(gpu, self.rank))
        else:
            self.use_gpu = False

        #>>>>>>>>>>>>>>>>> decide whether to use pretrained weights
        decide_checkpoints(cfg, keys=['point'])

        self.modules   = {}
        self.nets_dict = {}
        self.optimizers_dict = {}
        self.schedulers_dict = {}
        self.crit = []
        self.total_loss_dict = {}
        self.total_loss_dict['total_loss'] = 0
        self.total_loss_dict['handvertices_loss'] = 0
        self.total_loss_dict['handjoints_loss'] = 0
        self.total_loss_dict['regressionR_loss'] = 0
        self.total_loss_dict['regressionT_loss'] = 0 # extra
        self.total_loss_dict['objectness_loss'] = 0
        self.total_loss_dict['vote_loss'] = 0
        self.total_loss_dict['center_loss'] = 0
        self.total_loss_dict['center_loss1'] = 0
        self.total_loss_dict['center_loss2'] = 0
        self.total_loss_dict['confidence_loss'] = 0
        for head_name in cfg.HEAD.keys():
            if 'confidence' in head_name or 'mask' in head_name:
                continue
            self.total_loss_dict[head_name.split('_')[0] + '_loss'] = 0

        for key in ['point']:
            net_encoder = ModelBuilder.build_encoder(params=cfg.MODEL, weights=cfg[key].encoder_weights)
            net_decoder = ModelBuilder.build_decoder(params=cfg.MODEL, options=cfg.models[cfg.name_model], weights=cfg[key].decoder_weights)
            net_header, head_names  = ModelBuilder.build_header(layer_specs=cfg.HEAD, weights=cfg[key].header_weights)

            self.modules[key] = EncoderDecoder(net_encoder, net_decoder, net_header, head_names=head_names, cfg=cfg) # 2 crit
            weights_total= sum(p.numel() for p in self.modules[key].parameters())
            weights_grad = sum(p.numel() for p in self.modules[key].parameters() if p.requires_grad)

            print("Total number of parameters: ", weights_total)
            print("Total number of parameters requires_grad: ", weights_grad)
            if self.use_gpu:
                self.modules[key].cuda(gpu)
            self.nets_dict[key] = (net_encoder, net_decoder, net_header)

        if cfg.distributed:
            for key, node_backbone in self.modules.items():
                self.modules[key] = nn.parallel.DistributedDataParallel(node_backbone, device_ids=[gpu], find_unused_parameters=True)

        # Set up optimizers_dict
        for key, nets in self.nets_dict.items():
            optimizers = self.create_optimizers(nets, cfg)
            if cfg.TRAIN.lr_warmup:
                schedulers = self.create_schedulers(optimizers, cfg)
            else:
                schedulers = None
            self.schedulers_dict[key] = schedulers
            self.optimizers_dict[key] = optimizers

        if not cfg.TRAIN.lr_warmup:
            self.schedulers_dict =  None

    def reset(self):
        pass

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

            inputs = {'point_clouds': batch['P']}
            end_points = model(inputs=inputs)

            # pred_dict = model(batch['P'])

            if cfg.pred_mano:
                with torch.no_grad():
                    handvertices, handjoints  = model.mano_layer_gt(th_pose_coeffs=batch['regression_params'])
                    batch['handvertices']     = handvertices/200
                    batch['handjoints']       = handjoints/200

            for key in batch:
                assert(key not in end_points)
                end_points[key] = batch[key]

            loss_total, end_points = get_loss(end_points, cfg)
            # config loss_dict
            self.total_loss_dict['objectness_loss'] = end_points['objectness_loss']
            self.total_loss_dict['center_loss'] = end_points['center_loss']
            self.total_loss_dict['center_loss1'] = end_points['center_loss1']
            self.total_loss_dict['center_loss2'] = end_points['center_loss2']
            self.total_loss_dict['confidence_loss'] = end_points['confidence_loss']
            self.total_loss_dict['vote_loss'] = end_points['vote_loss']

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
                    adjust_learning_rate(optimizers, cfg) # adjust according to step
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

            print('Training: [{0}][{1}/{2}] | ''Lr: {lr_encoder:.3e} {lr_decoder:.3e} | '
                    'Loss: {loss:.3f} o:{objectness_loss:.3f} co:{confidence_loss:.3f} c:{center_loss:.3f} v:{vote_loss:.3f}'.format(epoch, i, len(loader), \
                      loss=loss_total, \
                      objectness_loss=self.total_loss_dict['objectness_loss'], \
                      center_loss=self.total_loss_dict['center_loss'], \
                      vote_loss=self.total_loss_dict['vote_loss'], \
                      confidence_loss=self.total_loss_dict['confidence_loss'], \
                      lr_encoder=cfg.TRAIN.running_lr_encoder, \
                      lr_decoder=cfg.TRAIN.running_lr_decoder))

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

                inputs = {'point_clouds': batch['P']}
                end_points = model(inputs=inputs)
                if cfg.pred_mano:
                    with torch.no_grad():
                        handvertices, handjoints = model.mano_layer_gt(th_pose_coeffs=batch['regression_params'])
                        batch['handvertices']     = handvertices/200
                        batch['handjoints']       = handjoints/200

                for key in batch:
                    assert(key not in end_points)
                    end_points[key] = batch[key]

                loss_total, end_points = get_loss(end_points, cfg)
                # config loss_dict
                self.total_loss_dict['objectness_loss'] = end_points['objectness_loss']
                self.total_loss_dict['center_loss'] = end_points['center_loss']
                self.total_loss_dict['vote_loss'] = end_points['vote_loss']
                self.total_loss_dict['confidence_loss'] = end_points['confidence_loss']
                # # combine different losses
                # loss_total= self.collect_losses(loss_dict)
                loss_total = loss_total.mean()

                for key, loss_item in self.total_loss_dict.items():
                    if isinstance(loss_item, int):
                        continue
                    losses_meter[key].update(loss_item.item(), 1)
                batch_time.update(time.time() - tic)

                print('Validation: [{0}][{1}/{2}] | '
                    'Loss: {loss:.3f} o:{objectness_loss:.3f} co:{confidence_loss:.3f} c:{center_loss:.3f} v:{vote_loss:.3f}'.format(epoch, i, len(loader), \
                          loss=loss_total, \
                          objectness_loss=self.total_loss_dict['objectness_loss'], \
                          confidence_loss=self.total_loss_dict['confidence_loss'], \
                          center_loss=self.total_loss_dict['center_loss'], \
                          vote_loss=self.total_loss_dict['vote_loss']))
                # # >>>>>>>>>>>>> logging
                # print('---Validation: [{0}][{1}/{2}] | '
                #     'Loss: {loss:.3f} m:{miou_loss:.3f} n:{nocs_loss:.3f} h:{heatmap_loss:.3f} u:{unitvec_loss:.3f} r:{regression_loss:.3f}, R:{regressionR_loss:.3f}, hv: {handvertices_loss:.3f}, hj: {handjoints_loss:.3f}'.format(epoch, i, len(loader), \
                #       loss=loss_total, \
                #       miou_loss=self.total_loss_dict['partcls_loss'], \
                #       nocs_loss=self.total_loss_dict['nocs_loss'], \
                #       regression_loss=self.total_loss_dict['regression_loss'],\
                #       regressionR_loss=self.total_loss_dict['regressionR_loss'],\
                #       handvertices_loss=self.total_loss_dict['handvertices_loss'],\
                #       handjoints_loss=self.total_loss_dict['handjoints_loss'],\
                #       heatmap_loss=self.total_loss_dict['handheatmap_loss'],\
                #       unitvec_loss=self.total_loss_dict['handunitvec_loss'],\
                #       ))
                # >>>>>>>>>>>>> save
                pred_dict = {}
                for key, value in end_points.items():
                    if len(value.size())>1:
                        pred_dict[key] = value
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

    # as we have multiple losses funcs, here we try to combine each one
    def collect_losses(self, loss_dict):
        self.total_loss = 0
        for key, content in loss_dict.items():
            self.total_loss_dict[key.split('_')[0] + '_loss'] = torch.mean(loss_dict[key])

            # decide not added to total loss
            if 'hand' in key and not self.pred_hand:
                continue
            multiplier_factor = self.config.TRAIN[key.split('_')[0] + '_loss_multiplier']
            self.total_loss += multiplier_factor * self.total_loss_dict[key.split('_')[0] + '_loss']
            if self.config.verbose:
                print('collecting loss for ', key, self.total_loss_dict[key.split('_')[0] + '_loss'].detach().cpu().numpy(), multiplier_factor)
        self.total_loss *= self.config.TRAIN.total_loss_multiplier
        self.total_loss_dict['total_loss'] = self.total_loss

        return self.total_loss

    # use dictionary to
    def fill_gpu_batch_data(self, gpu_id, batch):
        for key, item in batch.items():
            batch[key] = batch[key].cuda(gpu_id, non_blocking=True)

    # scomp
    def compute_loss(self, pred_dict, gt_dict, cfg):
        '''
            Input:
                pred_dict should contain:
                    - part_cls_per_point: BxKxN, segmentation parts;
                    - nocs_per_point: Bx3xN, nocs per point

        '''
        # dimension tensors
        n_max_parts= cfg.n_max_parts
        loss_dict= {}

        for key, value in pred_dict.items():
            if 'confidence' in key or 'mask' in key:
                continue
            elif 'regression' in key:
                if key == 'regression_params':
                    diff_p = (pred_dict[key] - gt_dict[key][:, 6:]).view(value.size(0), -1, 3).contiguous()
                    loss_dict[key] = torch.mean(torch.norm(diff_p, dim=2), dim=1)
                elif key=='regressionR':
                    loss_dict[key] = loss_geodesic(pred_dict[key], gt_dict[key])
                else:
                    # regressionT
                    diff_p = (pred_dict[key] - gt_dict[key]).view(value.size(0), -1, 3).contiguous()
                    loss_dict[key] = torch.mean(torch.norm(diff_p, dim=2))
                    # loss_dict[key] = torch.mean(torch.norm(diff_p, dim=2), dim=1)
            elif 'cls' in key:
                loss_dict[key] = compute_miou_loss(pred_dict[key], gt_dict[key], loss_type=cfg.TRAIN.loss)
            elif 'nocs' in key:
                loss_dict[key] = compute_nocs_loss(pred_dict[key], gt_dict[key], confidence=None, \
                                                num_parts=n_max_parts, mask_array=gt_dict['part_mask'], MULTI_HEAD=True, SELF_SU=False) # todo
            elif 'gocs' in key:
                loss_dict[key] = compute_nocs_loss(pred_dict[key], gt_dict[key], confidence=None, \
                                        num_parts=n_max_parts, MULTI_HEAD=False, SELF_SU=False)
            elif 'hand' in key:
                if key=='handheatmap_per_point' or key=='handunitvec_per_point': #
                    loss_dict[key] = compute_multi_offsets_loss(pred_dict[key], gt_dict[key], confidence=gt_dict['hand_mask'])
                else:
                    # print('mean norm for ', key, pred_dict[key].shape, gt_dict[key].shape)
                    diff_p = (pred_dict[key] - gt_dict[key]) # N, 3
                    loss_dict[key] = torch.mean(torch.norm(diff_p, dim=2), dim=1)
            else:
                print('compute vect loss for ', key)
                loss_dict[key] = compute_vect_loss(pred_dict[key], gt_dict[key], confidence=gt_dict['joint_mask'], TYPE_LOSS='L2')

        return loss_dict

    def checkpoint(self, nets, cfg, epoch, suffix):
        print('Saving checkpoints...', epoch)
        (net_encoder, net_decoder, head) = nets
        if net_encoder:
            dict_encoder = net_encoder.state_dict()
            torch.save(
                dict_encoder,
                '{}/encoder_best_{}.pth'.format(cfg.log_dir, suffix))
        if net_decoder:
            dict_decoder = net_decoder.state_dict()
            torch.save(
                dict_decoder,
                '{}/decoder_best_{}.pth'.format(cfg.log_dir, suffix))
        if head:
            dict_head = head.state_dict()
            torch.save(
                dict_head,
                '{}/head_best_{}.pth'.format(cfg.log_dir, suffix))

    def get_crit(self, num_classes_dict, cfg):
        parser    = self.parser
        DATA      = self.DATA

        epsilon_w = cfg.TRAIN.epsilon_w
        content   = torch.zeros(num_classes_dict['c'], dtype=torch.float)
        content_m = torch.zeros(num_classes_dict['m'], dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = parser.to_xentropy(cl)  # map actual class to xentropy class
            x_mo = parser.to_motion(cl)
            content[x_cl]   += freq
            content_m[x_mo] += freq

        loss_w = {}
        loss_w['c'] = 1 / (content + epsilon_w)   # get weights
        loss_w['m'] = 1 / (content_m + epsilon_w)

        loss_w['m'][0] = 0
        for x_cl, w in enumerate(loss_w['c']):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                loss_w['c'][x_cl] = 0
        print("Loss weights from content of segmentation: \n", loss_w['c'].data)
        print("Loss weights from content of motion: \n", loss_w['m'].data)

        if cfg.TRAIN.loss == "xentropy":
            print('using xentropy loss for training')
            crit_c = nn.NLLLoss(weight=loss_w['c'], ignore_index=-1)
            crit_m = nn.NLLLoss(weight=loss_w['m'], ignore_index=-1)
        elif cfg.TRAIN.loss == "focalloss":
            print('using focal loss for training')
            crit_c = FocalLoss(gamma=2.0, weight=loss_w['c'], ignore_index=-1)
            crit_m = FocalLoss(gamma=2.0, weight=loss_w['m'], ignore_index=-1)
        crit = {'c': crit_c, 'm': crit_m}
        return crit, loss_w

    def group_weight(self, module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def create_optimizers(self, nets, cfg):
        (net_encoder, net_decoder, head) = nets
        optimizers = [None] * 3
        if net_encoder:
            optimizer_encoder = torch.optim.SGD(
                net_encoder.parameters(),
                lr=cfg.TRAIN.lr_encoder,
                momentum=cfg.TRAIN.beta1,
                weight_decay=cfg.TRAIN.weight_decay)
            optimizers[0] = optimizer_encoder
        if net_decoder:
            optimizer_decoder = torch.optim.SGD(
                net_decoder.parameters(),
                lr=cfg.TRAIN.lr_decoder,
                momentum=cfg.TRAIN.beta1,
                weight_decay=cfg.TRAIN.weight_decay)
            optimizers[1] = optimizer_decoder
        if head:
            optimizer_decoder = torch.optim.SGD(
                head.parameters(),
                lr=cfg.TRAIN.lr_decoder,
                momentum=cfg.TRAIN.beta1,
                weight_decay=cfg.TRAIN.weight_decay)
            optimizers[2] = optimizer_decoder
        return tuple(optimizers)

    def create_schedulers(self, optimizers, cfg):
        # get schedulers
        schedulers   = []
        lr_list      = [cfg.TRAIN.lr_encoder, cfg.TRAIN.lr_decoder, cfg.TRAIN.lr_decoder]
        warmup_steps = cfg.TRAIN.epoch_iters * cfg.TRAIN.warmup_epochs
        final_decay  = cfg.TRAIN.lr_decay ** (1/cfg.TRAIN.epoch_iters)

        for i, optimizer in enumerate(optimizers):
            if optimizer is not None:
                scheduler = warmupLR(optimizer=optimizer,
                                      lr=lr_list[i],
                                      warmup_steps=warmup_steps,
                                      momentum=cfg.TRAIN.momentum,
                                      decay=final_decay)
                schedulers.append(scheduler)
            else:
                schedulers.append(None)

        return schedulers

    def create_evaluator(self, gpu_id, loss_w_dict, num_classes_dict):
        ignore_class = {}
        evaluator  = {}
        run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('current device: ', torch.cuda.current_device())

        for key, w_loss in loss_w_dict.items():
            ignore_class[key] = []
            for i, w in enumerate(w_loss):
              if w < 1e-10:
                ignore_class[key].append(i)
                print("Ignoring class ", i, " in IoU evaluation for ", key)
            evaluator[key]  = iouEval(num_classes_dict[key], torch.device('cpu'), ignore_class[key])

        return evaluator

    def adjust_learning_rate(self, optimizers, cfg):
        (optimizer_encoder, optimizer_decoder, optimizer_head) = optimizers
        if optimizer_encoder:
            for param_group in optimizer_encoder.param_groups:
                param_group['lr'] = cfg.TRAIN.running_lr_encoder
        if optimizer_decoder:
            for param_group in optimizer_decoder.param_groups:
                param_group['lr'] = cfg.TRAIN.running_lr_decoder
        if optimizer_head:
            for param_group in optimizer_head.param_groups:
                param_group['lr'] = cfg.TRAIN.running_lr_decoder
