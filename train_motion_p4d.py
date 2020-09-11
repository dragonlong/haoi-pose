# System libs
import os
import time
import math
import random
import imp
import yaml
import datetime
import argparse
import GPUtil
from distutils.version import LooseVersion

# Numerical libs
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
# Our libs
import __init__ as booger
from config import cfg
from modules.fuse_segmentator import *
from modules.mit_segmentator import ModelBuilder, SegmentationModuleFusion, EncoderDecoderModule
from common.train_utils import AverageMeter, parse_devices, setup_logger
from common.vis_utils import save_to_log, make_log_img
from common.losses import FocalLoss
from modules.ioueval import *
from common.warmupLR import *
from common.logger import Logger
from common.debugger import breakpoint

import dataset.kitti.parser_fusion_conv3d as parserModule

class SegmentationModule(nn.Module):
    def __init__(self, net_enc, net_dec, crit, crf=None, deep_sup_scale=None, loss_type=None):
        super(SegmentationModule, self).__init__()
        self.backbone = net_enc
        self.decoder  = net_dec
        self.head = None
        self.CRF  = crf
        self.crit_c = crit['c']
        self.crit_m = crit['m']
        self.loss_type = loss_type

    def forward(self, xyz1, xyz2, feat1, feat2, y=None, mask=None, segSize=None, index=0):
        if self.backbone is None:
            pred = self.decoder(xyz1, xyz2, feat2)
        else:
            pred = self.decoder(self.backbone(xyz1, return_feature_maps=True))

        # loss part
        motion_pred = pred
        class_pred  = motion_pred
        cls_mask, mot_mask  = mask
        y_c, y_m = y
        if cls_mask is not None:
            pred = class_pred * cls_mask
        if mot_mask is not None:
            # breakpoint()
            pred = motion_pred * mot_mask

        if y is not None: # train, need loss
            if self.loss_type == 'focalloss':
                # loss_c = self.crit(class_pred, y_c)
                loss_c = torch.Tensor([0])
                loss_m = self.crit(motion_pred, y_m)
            elif self.loss_type == 'xentropy':
                # loss_c = self.crit_c(nn.functional.log_softmax(class_pred, dim=1), y_c)
                loss_c = torch.Tensor([0])
                loss_m = self.crit_m(nn.functional.log_softmax(motion_pred, dim=1), y_m)
        else:
            loss_c, loss_m = None, None
        loss_dict = {'c': loss_c, 'm': loss_m}
        pred_dict = {'c': class_pred, 'm': motion_pred}
        return loss_dict, pred_dict

def train_epoch(gpu_id, modules, segmentation_module, train_loader, optimizers_dict, evaluator_dict, epoch, cfg, schedulers_dict=None):
    batch_time = AverageMeter()
    data_time  = AverageMeter()

    losses     = AverageMeter()
    losses_c   = AverageMeter()
    losses_m   = AverageMeter()

    acc_c        = AverageMeter()
    iou_c        = AverageMeter()

    acc_m        = AverageMeter()
    iou_m        = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)
    torch.cuda.empty_cache()
    # main loop
    tic = time.time()
    # padded_input, non_empty_map, pixel_cat_map, motion_ignore_map, motion_state_map
    for i, (xyz1, xyz2, feat1, feat2, motion_mask, unproj_labels, motion_labels) in enumerate(train_loader):
        #if i >25:
        #    break
        # breakpoint()
        # save_viz = True
        # if save_viz:
        #     viz_dict = {}
        #     print('!!! Saving visualization data')
        #     viz_dict['input'] = p_feat.cpu().numpy()
        #     viz_dict['label'] = proj_labels.cpu().numpy()
        #     viz_dict['coord'] = coords.cpu().numpy()
        #     viz_dict['motion'] = motion_labels.cpu().numpy()
        #     viz_dict['mask'] = proj_mask.cpu().numpy()
        #     viz_dict['m_mask']=motion_mask.cpu().numpy()

        #     for key, value in viz_dict.items():
        #         print(key, value.shape)
        #     save_name = f'/home/lxiaol9/log/test/{i}_viz_data.npy'
        #     print('saving to ', save_name)
        #     np.save(save_name, arr=viz_dict)

        xyz1 = xyz1.cuda(gpu_id, non_blocking=True)
        xyz2 = xyz2.cuda(gpu_id, non_blocking=True) 
        feat1= feat1.cuda(gpu_id, non_blocking=True)
        feat2= feat2.cuda(gpu_id, non_blocking=True)
        in_vol = xyz1
        proj_mask   = None
        proj_labels = unproj_labels.cuda(gpu_id, non_blocking=True).long()
        motion_mask = motion_mask.cuda(gpu_id, non_blocking=True)
        motion_labels = motion_labels.cuda(gpu_id, non_blocking=True).long()
        data_time.update(time.time() - tic)

        segmentation_module.zero_grad()
        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters

        loss_dict, pred_dict = segmentation_module(xyz1, xyz2, feat1, feat2, [proj_labels, motion_labels], mask=[proj_mask, motion_mask], index=i)
        loss_total = loss_dict['m']
        loss_total.backward()
        #
        for _, optimizers in optimizers_dict.items():
            for optimizer in optimizers:
                if optimizer:
                    optimizer.step()

        loss_total = loss_total.mean()
        loss_c = loss_dict['c'].mean()
        loss_m = loss_dict['m'].mean()

        output = [pred_dict['c'], pred_dict['m']]
        evaluator = [evaluator_dict['c'], evaluator_dict['m']]
        labels = [proj_labels, motion_labels]
        accs   = [acc_c, acc_m]
        ious   = [iou_c, iou_m]
        argmaxs= []
        with torch.no_grad():
            for k in range(2):
                evaluator[k].reset()
                argmax = output[k].argmax(dim=1)
                argmaxs.append(argmax)
                evaluator[k].addBatch(argmax.type(torch.long), labels[k].type(torch.long))
                accuracy = evaluator[k].getacc()
                jaccard, class_jaccard = evaluator[k].getIoU()
                accs[k].update(accuracy.item(), in_vol.size(0))
                ious[k].update(jaccard.item(), in_vol.size(0))

        losses.update(loss_total.item(), in_vol.size(0))
        losses_c.update(loss_c.item(), in_vol.size(0))
        losses_m.update(loss_m.item(), in_vol.size(0))

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        if schedulers_dict is None:
            scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
            cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
            cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr
            for _, optimizers in optimizers_dict.items():
                adjust_learning_rate(optimizers, cfg) # adjust according to step
        else:
            schedulers = schedulers_dict[list(schedulers_dict.keys())[0]]
            for j, scheduler in enumerate(schedulers):
                if scheduler is not None:
                    scheduler.step()
            if optimizers[0] is not None:
                for g in optimizers[0].param_groups:
                    cfg.TRAIN.running_lr_encoder = g["lr"]
            if optimizers[1] is not None:
                for g in optimizers[1].param_groups:
                    cfg.TRAIN.running_lr_decoder = g["lr"]

        # calculate accuracy, and display
        if gpu_id<2:
            print('Lr: {lr_encoder:.3e} {lr_decoder:.3e} | '
                  'Epoch: [{0}][{1}/{2}] | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'CLoss {loss_c.val:.4f} ({loss_c.avg:.4f}) | '
                  'MLoss {loss_m.val:.4f} ({loss_m.avg:.4f}) | '
                  'Cacc {acc[0].val:.3f} ({acc[0].avg:.3f}) | '
                  'CIoU {iou[0].val:.3f} ({iou[0].avg:.3f}) | '
                  'Macc {acc[1].val:.3f} ({acc[1].avg:.3f}) | '
                  'MIoU {iou[1].val:.3f} ({iou[1].avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, loss_c=losses_c, loss_m=losses_m, acc=accs, iou=ious,
                      lr_encoder=cfg.TRAIN.running_lr_encoder,
                      lr_decoder=cfg.TRAIN.running_lr_decoder))

    acc_avg = [accs[0].avg, accs[1].avg]
    iou_avg = [ious[0].avg, ious[1].avg]
    losses_avg = [losses.avg, losses_c.avg, losses_m.avg]

    return acc_avg, iou_avg, losses_avg

def validate_epoch(gpu_id, modules, segmentation_module, valid_loader, evaluator_dict, class_func, color_fn, save_scans):
    batch_time = AverageMeter()
    losses     = AverageMeter()
    losses_c   = AverageMeter()
    losses_m   = AverageMeter()

    acc_c        = AverageMeter()
    iou_c        = AverageMeter()

    acc_m        = AverageMeter()
    iou_m        = AverageMeter()
    rand_imgs  = []

    # switch to evaluate mode
    segmentation_module.eval()
    evaluator_dict['c'].reset()
    evaluator_dict['m'].reset()
    torch.cuda.empty_cache()

    # main loop,
    tic = time.time()
    with torch.no_grad():
        evaluator = [evaluator_dict['c'], evaluator_dict['m']]
        for i, (xyz1, xyz2, feat1, feat2, motion_mask, unproj_labels, motion_labels) in enumerate(valid_loader):
            # if i > 1500:
            #    break

            pts = xyz1.cpu().numpy()
            pts = pts[:, :3, :].transpose(0, 2, 1).reshape(-1, 3)
            extents = np.array([[-48., 48.], [-48., 48.], [-3.2, 3.2]]).astype(np.float32)
            filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                                  (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                                  (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]

            xyz1 = xyz1.cuda(gpu_id, non_blocking=True)
            xyz2 = xyz2.cuda(gpu_id, non_blocking=True) 
            feat1= feat1.cuda(gpu_id, non_blocking=True)
            feat2= feat2.cuda(gpu_id, non_blocking=True)
            in_vol = xyz1
            proj_mask   = None
            proj_labels = unproj_labels.cuda(gpu_id, non_blocking=True).long()
            motion_mask = motion_mask.cuda(gpu_id, non_blocking=True)
            motion_labels = motion_labels.cuda(gpu_id, non_blocking=True).long()
            if i % 50 == 0:
                print('iterating over ', i)

            segmentation_module.zero_grad()
            loss_dict, pred_dict = segmentation_module(xyz1, xyz2, feat1, feat2, [proj_labels, motion_labels], mask=[proj_mask, motion_mask])
            loss_total = 1* loss_dict['m']
            loss_total = loss_total.mean()
            loss_c = loss_dict['c'].mean()
            loss_m = loss_dict['m'].mean()

            losses.update(loss_total.mean().item(), in_vol.size(0))
            losses_c.update(loss_c.mean(), in_vol.size(0))
            losses_m.update(loss_m.mean(), in_vol.size(0))


            output = [pred_dict['c'], pred_dict['m']]
            labels = [proj_labels, motion_labels]
            argmaxs= []
            for k in range(2):
                argmax = output[k].argmax(dim=1)
                argmaxs.append(argmax)
                argmax = argmax.cpu().numpy().reshape(-1)[filter_idx]
                label_cur = labels[k].cpu().numpy().reshape(-1)[filter_idx]
                evaluator[k].addBatch(argmax, label_cur)
                if k ==0:
                    if save_scans:
                        mask_np  = proj_mask[0].cpu().numpy()
                        depth_np = in_vol[0].cpu().numpy()
                        pred_np  = argmax[0].cpu().numpy()
                        gt_np    = unproj_labels[0].cpu().numpy()
                        out      = make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)
                        rand_imgs.append(out)

            batch_time.update(time.time() - tic)
            tic = time.time() # update on every iteration

        accs  = [acc_c, acc_m]
        ious  = [iou_c, iou_m]
        for k in range(2):
            accuracy = evaluator[k].getacc()
            jaccard, class_jaccard = evaluator[k].getIoU()
            accs[k].update(accuracy.item(), in_vol.size(0))
            ious[k].update(jaccard.item(), in_vol.size(0))
            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_func(i), jacc=jacc))

        print('Validation set:\n'
              'Time pb {batch_time.avg:.3f}\n'
              'loss{loss.avg:.4f}\n'
              'Closs {loss_c.avg:.4f}\n'
              'Mloss {loss_m.avg:.4f}\n'
              'CAcc: {acc[0].avg:.3f}\n'
              'CIoU {iou[0].avg:.3f}'
              'MAcc: {acc[1].avg:.3f}\n'
              'MIoU {iou[1].avg:.3f}'.format(batch_time=batch_time,
                                             loss=losses, loss_c=losses_c, loss_m=losses_m,
                                             acc=accs, iou=ious))

    acc_avg = [accs[0].avg, accs[1].avg]
    iou_avg = [ious[0].avg, ious[1].avg]
    losses_avg = [losses.avg, losses_c.avg, losses_m.avg]

    return acc_avg, iou_avg, losses_avg, rand_imgs

def checkpoint(nets, cfg, epoch, suffix):
    print('Saving checkpoints...', epoch)
    (net_encoder, net_decoder, head) = nets
    if net_encoder:
        dict_encoder = net_encoder.state_dict()
        torch.save(
            dict_encoder,
            '{}/encoder_best_{}.pth'.format(cfg.DIR, suffix))
    if net_decoder:
        dict_decoder = net_decoder.state_dict()
        torch.save(
            dict_decoder,
            '{}/decoder_best_{}.pth'.format(cfg.DIR, suffix))
    if head:
        dict_head = head.state_dict()
        torch.save(
            dict_head,
            '{}/head_best_{}.pth'.format(cfg.DIR, suffix))

def get_crit(num_classes, parser, cfg, DATA):
    epsilon_w = cfg.TRAIN.epsilon_w
    content   = torch.zeros(num_classes['c'], dtype=torch.float)
    content_m = torch.zeros(num_classes['m'], dtype=torch.float)
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

def group_weight(module):
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

def create_optimizers(nets, cfg):
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

def create_schedulers(optimizers, cfg):
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

def create_evaluator(gpu_id, loss_w_dict, num_classes_dict):
    ignore_class = {}
    evaluator = {}
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

def main_worker(gpu, gpus, cfg, DATA, RV_CFG, args=None):
    cfg.TRAIN.rank = 0 * len(gpus) + gpu
    gpu_id = gpu
    # gpu = gpus[gpu]

    print('using gpu id: {} with rank: {}'.format(gpu, cfg.TRAIN.rank))
    torch.cuda.set_device(gpu)
    if cfg.TRAIN.distributed:
        try:                                    # set up GPU
            dist.init_process_group(backend='nccl', init_method='env://', world_size=cfg.TRAIN.world_size, rank=cfg.TRAIN.rank)
        except:
            os.environ['MASTER_PORT'] = '8889'
            dist.init_process_group(backend='nccl', init_method='env://', world_size=cfg.TRAIN.world_size, rank=cfg.TRAIN.rank)
    batch_size  = cfg.TRAIN.batch_size_per_gpu
    in_channels  = 128    #TODO
    stack_factor = 14
    if cfg.BEV.label_dim == 2:
        stack_factor = 1
    num_worker  = int(cfg.TRAIN.workers/len(gpus))
    print('we have ', num_worker, ' worker per gpu process')
    # get the data
    parser = parserModule.Parser(root=cfg.DATASET.root_dataset,
                                  train_sequences=DATA["split"]["train"],
                                  valid_sequences=DATA["split"]["valid"],
                                  test_sequences=None,
                                  labels=DATA["labels"],
                                  color_map=DATA["color_map"],
                                  learning_map=DATA["learning_map"],
                                  learning_map_inv=DATA["learning_map_inv"],
                                  cfg=cfg,
                                  max_points=cfg.DATASET.max_points,
                                  batch_size=batch_size,
                                  workers=num_worker,
                                  gt=True,
                                  live_compute=True,
                                  shuffle_train=True)

    print('Getting data loader ready')
    num_classes    = parser.get_n_classes()
    train_loader   = parser.get_train_set()
    valid_loader   = parser.get_valid_set()

    # Network Builders
    backbones = {}
    modules   = {}
    nets_dict = {}
    optimizers_dict = {}
    schedulers_dict = {}
    crit = []
    backbones['bev'] = {'arch_encoder': cfg.BEV.arch_encoder.lower(),
                        'arch_decoder': cfg.BEV.arch_decoder.lower(),
                        'weights_encoder': cfg.BEV.weights_encoder,
                        'weights_decoder': cfg.BEV.weights_decoder,
                        'fc_dim': cfg.BEV.fc_dim }
    if args.eval: 
        # backbones['bev']['weights_encoder'] = '/homes/xili/log/point_motion_lr5-3_warm_xent_pointnet2_dpcn/encoder_best_valid.pth'
        backbones['bev']['weights_decoder'] = '/homes/xili/log/point_motion_lr5-3_warm_xent_pointnet2_dpcn/decoder_best_bev_val.pth'
    for key, backbone in backbones.items():
        # net enc & decoder
        net_encoder = ModelBuilder.build_encoder(
            arch=backbone['arch_encoder'],
            fc_dim=backbone['fc_dim'],
            weights=backbone['weights_encoder'])

        net_decoder = ModelBuilder.build_decoder(
            arch=backbone['arch_decoder'],
            fc_dim=backbone['fc_dim'],
            num_class=num_classes,
            in_channels=in_channels,
            stack_factor=stack_factor,
            depth=cfg.MODEL.depth_decoder,
            up_mode=cfg.MODEL.upmode_decoder,
            weights=backbone['weights_decoder'])
        num_classes_dict = {'c': num_classes, 'm': 3}

        crit_per_branch, loss_w_dict = get_crit(num_classes_dict, parser, cfg, DATA)
        modules[key] = SegmentationModule(net_encoder, net_decoder, crit_per_branch, loss_type=cfg.TRAIN.loss) # 2 crit
        weights_total = sum(p.numel() for p in modules[key].parameters())
        weights_grad = sum(p.numel() for p in modules[key].parameters() if p.requires_grad)
        print("Total number of parameters: ", weights_total)
        print("Total number of parameters requires_grad: ", weights_grad)
        modules[key].cuda(gpu)

        nets_dict[key] = (net_encoder, net_decoder, None)
        crit.append(crit_per_branch)

    if cfg.TRAIN.distributed:
        for key, node_backbone in modules.items():
            modules[key] = nn.parallel.DistributedDataParallel(node_backbone, device_ids=[gpu], find_unused_parameters=True)

    # Set up optimizers_dict
    for key, nets in nets_dict.items():
        optimizers = create_optimizers(nets, cfg)
        if cfg.TRAIN.lr_warmup:
            schedulers = create_schedulers(optimizers, cfg)
        else:
            schedulers = None
        schedulers_dict[key] = schedulers
        optimizers_dict[key] = optimizers
    if not cfg.TRAIN.lr_warmup:
        schedulers_dict =  None

    tb_logger = Logger(cfg.DIR + "/tb")
    info = {}
    # accuracy and IoU stuff
    best_train_iou = 0.0
    best_val_iou   = 0.0

    evaluator = create_evaluator(gpu, loss_w_dict, num_classes_dict) #for now, we just try to to device

    # main loop
    print('start training from ', cfg.TRAIN.start_epoch)
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        print('iterating over epoch {}'.format(epoch))
        if not args.eval:
            train_acc, train_iou, train_loss = train_epoch(gpu, None, modules['bev'], train_loader, optimizers_dict, evaluator, epoch+1, cfg, schedulers_dict)
            info["train_total_loss"]  = train_loss[0]
            info["train_loss"] = train_loss[1]
            info["train_Mloss"] = train_loss[2]

            info["train_acc"] = train_acc[0]
            info["train_iou"] = train_iou[0]
            info["train_Macc"] = train_acc[1]
            info["train_Miou"] = train_iou[1]
            if best_train_iou < train_iou[1] and (gpu_id==0 or gpu==0):
                for key, nets in nets_dict.items():
                    checkpoint(nets, cfg, epoch+1, suffix= key + '_train')
                best_train_iou = train_iou[1]
        val_acc, val_iou, val_loss, rand_imgs = validate_epoch(gpu, None, modules['bev'], valid_loader, evaluator, parser.get_xentropy_class_string, parser.to_color, save_scans=cfg.TRAIN.save_scans)
        # update info
        info["valid_total_loss"]  = val_loss[0]
        info["valid_loss"] = val_loss[1]
        info["valid_Mloss"] = val_loss[2]

        info["valid_acc"]  = val_acc[0]
        info["valid_iou"]  = val_iou[0]
        info["valid_Macc"] = val_acc[1]
        info["valid_Miou"] = val_iou[1]
        #breakpoint()
        if best_val_iou < val_iou[1] and (gpu_id==0 or gpu==0):
            for key, nets in nets_dict.items():
                checkpoint(nets, cfg, epoch+1, suffix= key + '_val')
            best_val_iou = val_iou[1]
        info['encoder_lr'] = cfg.TRAIN.running_lr_encoder
        info['decoder_lr'] = cfg.TRAIN.running_lr_decoder
        if gpu_id==0 or gpu==0:
            save_to_log(logdir=cfg.DIR, logger=tb_logger, info=info, epoch=epoch, img_summary=cfg.TRAIN.save_scans, imgs=rand_imgs)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parsers = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parsers.add_argument(
        "--cfg_train", '-ct', default="config/arch/3_kitti-point-motionnet3d.yaml", type=str,
        metavar="FILE",
        help="path to config file",
    )
    parsers.add_argument(
        "--cfg_data", '-cd', default="config/labels/semantic-kitti.yaml", type=str,
        metavar="FILE",
        help="path to config file",
    )
    parsers.add_argument(
        "--cfg_rv", '-cr', default="config/arch/fusenet21.yaml", type=str,
        metavar="FILE",
        help="path to config file",
    )
    parsers.add_argument(
        '--log', '-l', type=str, default=os.path.expanduser("~") + '/logs/test_fuse/',
        help='Directory to put the log data. Default: ~/logs/date+time',
    )
    parsers.add_argument(
        "--num_gpus", default=4, type=int,
        help="gpus to use"
    )
    parsers.add_argument(
        "--num_nodes", default=1, type=int,
        help="nodes to use"
    )
    parsers.add_argument('--distributed', action='store_true',
        help='whether use parallel training')

    parsers.add_argument('--test', action='store_true',
        help='decide whether to test')

    parsers.add_argument('--eval', action='store_true',
        help='decide whether to test')

    args = parsers.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8801'

    cfg.merge_from_file(args.cfg_train)
    try:
        print("Opening data loader config file %s" % args.cfg_data)
        print("Opening rv backbone config file %s" % args.cfg_rv)
        DATA = yaml.safe_load(open(args.cfg_data, 'r'))
        RV_CFG = yaml.safe_load(open(args.cfg_rv, 'r'))
    except Exception as e:
        print(e)
        quit()
    # cfg.freeze()
    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loading data config: {}".format(args.cfg_train))
    logger.info("Running with:\n{}".format(args.cfg_train))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        assert os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = GPUtil.getAvailable(order = 'load', limit = min(int(args.num_gpus), torch.cuda.device_count()), maxLoad =0.2, maxMemory =0.2, includeNan=False, excludeID=[], excludeUUID=[])
    # gpus=[4, 5, 6, 7]
    print(f'available GPUs: {gpus}\n')
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size  = num_gpus * cfg.TRAIN.batch_size_per_gpu
    cfg.TRAIN.epoch_iters = int(20000 / cfg.TRAIN.batch_size)
    cfg.TRAIN.max_iters   = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder
    cfg.TRAIN.distributed        = args.distributed
    cfg.TRAIN.world_size         = num_gpus * args.num_nodes
    if args.test:
        cfg.TRAIN.mode           = 'test'
    cfg.DIR=args.log

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))
    #
    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)
    if args.distributed:
        mp.spawn(main_worker, nprocs=num_gpus, args=(gpus, cfg, DATA, RV_CFG, args))
    else:
        gpu = 0
        main_worker(gpu, gpus, cfg, DATA, RV_CFG, args)
