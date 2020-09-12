import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import __init__ as booger
from modules.mit_segmentator import ModelBuilder
from common.train_utils import AverageMeter, parse_devices, warmupLR
from common.vis_utils import save_to_log, make_log_img
from modules.losses import FocalLoss, compute_miou_loss, compute_nocs_loss, compute_vect_loss
from modules.ioueval import *
from common.logger import Logger

def breakpoint():
    import pdb;pdb.set_trace()

class EncoderDecoder(nn.Module):
    def __init__(self, net_enc, net_dec, head=None, head_names=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = net_enc
        self.decoder  = net_dec
        self.head     = head
        self.head_names= head_names

    def forward(self, xyz1, mask=None, index=0):
        if self.backbone is None:
            net = self.decoder(xyz1)
        else:
            net = self.decoder(self.backbone(xyz1))
        pred_dict = {}
        # safe
        for i, sub_head in enumerate(self.head):
            pred_dict[self.head_names[i]] = self.head[i](net)

        return pred_dict

class Network():
    def __init__(self, gpu, cfg): # with seperate gpu for distributed training
        self.rank = gpu
        self.config  = cfg
        self.use_gpu = True
        self.is_mixed  = False
        self.pred_joint= cfg.pred_joint
        self.pred_joint_ind = cfg.pred_joint_ind
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

        self.modules   = {}
        self.nets_dict = {}
        self.optimizers_dict = {}
        self.schedulers_dict = {}
        self.crit = []

        for key in ['point']:
            net_encoder = ModelBuilder.build_encoder(params=cfg.MODEL)
            net_decoder = ModelBuilder.build_decoder(params=cfg.MODEL, options=cfg.models[cfg.name_model])
            net_header, head_names  = ModelBuilder.build_header(layer_specs=cfg.HEAD)

            self.modules[key] = EncoderDecoder(net_encoder, net_decoder, net_header, head_names=head_names) # 2 crit

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

    def train_epoch(self, gpu_id, loader, model, epoch, cfg, modules=None):
        """
        if we have pretrained features, we could use modules
        """
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        losses     = AverageMeter()

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
            loss_dict = self.compute_loss(pred_dict, batch)

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
            loss_total = loss_total.mean()
            losses.update(loss_total.item(), batch_size)
            batch_time.update(time.time() - tic)

            print('Lr: {lr_encoder:.3e} {lr_decoder:.3e} | '
                  'Epoch: [{0}][{1}/{2}] | '.format(epoch, i, len(loader), lr_encoder=cfg.TRAIN.running_lr_encoder, \
                      lr_decoder=cfg.TRAIN.running_lr_decoder))
        infos = {}


        return infos

    # as we have multiple losses funcs, here we try to combine each one
    def collect_losses(self, loss_dict):
        """
        confidence map is B*N*1
        """
        self.total_loss = 0

        nocs_loss_multiplier    = self.config.TRAIN.nocs_loss_multiplier
        heatmap_loss_multiplier = self.config.TRAIN.offset_loss_multiplier
        unitvec_loss_multiplier = self.config.TRAIN.offset_loss_multiplier
        orient_loss_multiplier  = self.config.TRAIN.orient_loss_multiplier
        gocs_loss_multiplier    = self.config.TRAIN.gocs_loss_multiplier
        index_loss_multiplier   = self.config.TRAIN.index_loss_multiplier
        miou_loss_multiplier    = self.config.TRAIN.miou_loss_multiplier

        self.total_nocs_loss = torch.mean(loss_dict['nocs_per_point'])
        self.total_miou_loss = torch.mean(loss_dict['part_cls_per_point'])

        # loss from heatmap estimation & offset estimation
        self.total_heatmap_loss = torch.mean(loss_dict['heatmap_per_point'])
        self.total_unitvec_loss = torch.mean(loss_dict['unitvec_per_point'])
        self.total_orient_loss  = torch.mean(loss_dict['orient_per_point'])
        self.total_index_loss   = torch.mean(loss_dict['joint_cls_per_point'])

        self.total_loss += nocs_loss_multiplier * self.total_nocs_loss
        self.total_loss += miou_loss_multiplier * self.total_miou_loss

        if self.is_mixed:
            self.total_gocs_loss = torch.mean(loss_dict['gocs_per_point'])
            self.total_loss += gocs_loss_multiplier * self.total_gocs_loss

        if self.pred_joint:   # todo
            if self.is_mixed: # only use it in part + global NOCS
                self.total_loss += heatmap_loss_multiplier * self.total_heatmap_loss
                self.total_loss += unitvec_loss_multiplier * self.total_unitvec_loss
            self.total_loss += orient_loss_multiplier * self.total_orient_loss # on joint axis
            if self.pred_joint_ind: # come with joint points association
                self.total_loss += index_loss_multiplier * self.total_index_loss

        self.total_loss *= self.config.TRAIN.total_loss_multiplier

        return self.total_loss

    # use dictionary to
    def fill_gpu_batch_data(self, gpu_id, batch):
        for key, item in batch.items():
            batch[key] = batch[key].cuda(gpu_id, non_blocking=True)

    def compute_loss(self, pred_dict, gt_dict):
        '''
            Input:
                pred_dict should contain:
                    - W: BxKxN, segmentation parts;
                    - nocs_per_point: Bx3xN, nocs per point

        '''
        # dimension tensors
        W          = pred_dict['part_cls_per_point']
        batch_size = W.size(0)   # B*N*K(k parts)
        n_max_parts= W.size(1)
        n_points   = W.size(2)

        confidence = pred_dict['confidence_per_point']
        loss_dict= {}

        for key, value in pred_dict.items():
            print(key, value.shape)
            if 'confidence' in key or 'mask' in key:
                continue
            elif key == 'W' or 'cls' in key:
                loss_dict[key] = compute_miou_loss(pred_dict[key], gt_dict[key])
            elif 'nocs' in key:
                loss_dict[key] = compute_nocs_loss(pred_dict[key], gt_dict[key], confidence=None, \
                                                num_parts=n_max_parts, mask_array=gt_dict['part_mask'], MULTI_HEAD=True, SELF_SU=False) # todo
            elif 'gocs' in key:
                loss_dict[key] = compute_nocs_loss(pred_dict[key], gt_dict[key], confidence=None, \
                                        num_parts=n_max_parts, MULTI_HEAD=False, SELF_SU=False)
            elif 'hand' in key:
                breakpoint()
                loss_dict[key] = compute_vect_loss(pred_dict[key], gt_dict[key], confidence=gt_dict['hand_mask'], TYPE_LOSS='L2')
            else:
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

    def adjust_learning_rate(optimizers, cfg):
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

    def train(self, cfg):
        tb_logger = Logger(cfg.log_dir + "/tb")
        info = {}
        # accuracy and IoU stuff
        best_train_iou = 0.0
        best_val_iou   = 0.0

        print('start training from ', cfg.TRAIN.start_epoch)
        for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
            print('iterating over epoch {}'.format(epoch))
            if not cfg.eval:
                train_acc, train_iou, train_loss = self.train_epoch(self.rank, None, self.modules['point'], epoch, cfg)
                info["train_total_loss"]  = train_loss[0]
                info["train_loss"] = train_loss[1]
                info["train_Mloss"] = train_loss[2]

                info["train_acc"] = train_acc[0]
                info["train_iou"] = train_iou[0]
                info["train_Macc"] = train_acc[1]
                info["train_Miou"] = train_iou[1]
                if best_train_iou < train_iou[1] and self.rank==0 and not cfg.eval:
                    for key, nets in self.nets_dict.items():
                        self.checkpoint(nets, cfg, epoch+1, suffix= key + '_train')
                    best_train_iou = train_iou[1]

            val_acc, val_iou, val_loss, rand_imgs = self.validate_epoch(self.rank, None, self.modules['point'], epoch, cfg, self.parser.get_xentropy_class_string, self.parser.to_color, save_scans=cfg.TRAIN.save_scans)

            # update info
            info["valid_total_loss"]  = val_loss[0]
            info["valid_loss"]  = val_loss[1]
            info["valid_Mloss"] = val_loss[2]

            info["valid_acc"]  = val_acc[0]
            info["valid_iou"]  = val_iou[0]
            info["valid_Macc"] = val_acc[1]
            info["valid_Miou"] = val_iou[1]

            #breakpoint()
            if best_val_iou < val_iou[1] and self.rank==0 and not cfg.eval:
                for key, nets in self.nets_dict.items():
                    self.checkpoint(nets, cfg, epoch+1, suffix= key + '_val')
                best_val_iou = val_iou[1]
            info['encoder_lr'] = cfg.TRAIN.running_lr_encoder
            info['decoder_lr'] = cfg.TRAIN.running_lr_decoder
            save_to_log(logdir=cfg.log_dir, logger=tb_logger, info=info, epoch=epoch, img_summary=cfg.TRAIN.save_scans, imgs=rand_imgs)

        print('Training Done!')
