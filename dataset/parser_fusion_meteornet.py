"""
Log:
4.20: reformulize it to point-voxel fusion

"""

import os
import yaml
import numpy as np
import time
import math
import h5py
import torch
from torch_scatter import scatter
from torch.utils.data import Dataset
from random import randint, sample
from numpy.linalg import inv
import matplotlib
# matplotlib.use('Agg') # TkAgg
import matplotlib.pyplot as plt  #
from multiprocessing import Manager

import __init__ as booger
from common.laserscan import LaserScan, SemLaserScan
from common.voxelvis import VoxelVis
from common.data_utils import voxelize_occupy, gen_2d_grid_gt_from_pts, gen_3d_voxel_gt_from_pts, save_h5_bev, voxelize_occupy_detail, parse_calibration, parse_poses
from dataset.kitti.base import PointCloudDataset, plot_imgs, plot_imgsup, is_scan, is_label, is_npy
from modules.ioueval import iouEval
EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL= ['.label']
EXTENSIONS_IND  = ['.npy']

import global_info as infos

exp_name   =infos.exp_name
data_dir   =infos.data_dir
model_dir               = infos.model_dir
suffix_kitti_dir        = infos.suffix_kitti_dir
suffix_motion_pred      = infos.suffix_motion_pred
suffix_submission_pred  = infos.suffix_submission_pred
suffix_rv_pred          = infos.suffix_rv_pred
suffix_bev_model_dir    = infos.suffix_bev_model_dir
suffix_rv_model_dir     = infos.suffix_rv_model_dir

# sssem
class SemanticKitti(PointCloudDataset):
  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               cfg,              # sensor to parse scans from
               max_points=150000, # max number of points present in dataset
               past_frame_skip=1, # use for similar data skipping
               future_frame_skip=1, # use
               seq_length=2,
               label_dim=2,   # dimension for the label
               gt=True,
               is_train=True,
               live_compute=False):      # send ground truth?
    PointCloudDataset.__init__(self, root, sequences, labels, color_map, learning_map, learning_map_inv ,\
      cfg, max_points, past_frame_skip, future_frame_skip, label_dim, gt, is_train, live_compute)

  def __getitem__(self, idx, verbose=False):
    if self.compute_on_the_fly:
      data = self.getitem_on_the_fly(idx, return_raw=True, return_test=True)
      xyz_list, feat_list, motion_mask, unproj_labels, unproj_motion_state, index_seq, index_frame, num_points, chosen_index = data
      xyzs = torch.cat(xyz_list, dim=-1)
      feats= torch.cat(feat_list, dim=-1)
      times= torch.ones(self.nframe, self.npoints).float() * torch.arange(0, self.nframe).unsqueeze(-1)
      times= times.view(1, -1).contiguous()
      if verbose:
        print(f'xyzs: {xyzs.size()}, times: {times.size()}, feats: {feats.size()}, motion_mask: {motion_mask.size()}')

      return xyz_list[0], xyzs, times, feats, unproj_labels, unproj_motion_state, motion_mask, index_seq, index_frame, num_points, chosen_index

class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               cfg,               # training cfg
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True,
               live_compute=False):  # shuffle training set?
    super(Parser, self).__init__()

    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.cfg = cfg
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = min(len(self.learning_map_inv), 20)
    if cfg.DATASET.augment:
        print('we are using data augmentation!!')
    # Data loading code
    if self.train_sequences:
      self.train_dataset = SemanticKitti(root=self.root,
                                         sequences=self.train_sequences,
                                         labels=self.labels,
                                         color_map=self.color_map,
                                         learning_map=self.learning_map,
                                         learning_map_inv=self.learning_map_inv,
                                         cfg=self.cfg,
                                         max_points=max_points,
                                         gt=self.gt,
                                         live_compute=live_compute
                                         )
      self.train_sampler = None
      if cfg.TRAIN.distributed:
        print('using training sampler with worlde size, and rank ', cfg.TRAIN.rank, cfg.TRAIN.world_size)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,
                                                                      num_replicas=cfg.TRAIN.world_size,
                                                                      rank=cfg.TRAIN.rank)

      self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=(self.shuffle_train and self.train_sampler is None),
                                                     num_workers=self.workers,
                                                     pin_memory=True,
                                                     sampler=self.train_sampler,
                                                     drop_last=True)
      assert len(self.trainloader) > 0
      self.trainiter = iter(self.trainloader)

    self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       cfg=self.cfg,
                                       max_points=max_points,
                                       gt=self.gt,
                                       is_train=False,
                                       live_compute=live_compute)
    # self.valid_sampler = None
    # if cfg.TRAIN.distributed:
    #   print('using training sampler')
    #   self.valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset,
    #                                                                 num_replicas=cfg.TRAIN.world_size,
    #                                                                 rank=cfg.TRAIN.rank)
    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)

    if self.test_sequences:
      self.test_dataset = SemanticKitti(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        cfg=self.cfg,
                                        max_points=max_points,
                                        gt=False,
                                        is_train=False,
                                        live_compute=live_compute)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_motion(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.valid_dataset.motion_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)

def breakpoint():
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    # all config args
    live_compute = True
    quick_test   = True
    is_verbose   = True
    save_flag    = False
    check_sanity = False
    check_input  = True
    check_bev    = False
    base_data    = data_dir + suffix_kitti_dir
    print('testing on the bev auto generation!')
    root = base_data
    DATA = yaml.safe_load(open('../../config/labels/semantic-kitti.yaml', 'r'))
    cfg.merge_from_file('../../config/arch/3_kitti-point-motionnet3d.yaml')
    train_sequences= DATA["split"]["train"]
    test_sequences = DATA["split"]["test"]
    val_sequences  = DATA["split"]["valid"]

    #
    labels=DATA["labels"]
    color_map=DATA["color_map"]
    learning_map=DATA["learning_map"]
    learning_map_inv=DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]

    max_points=cfg.DATASET.max_points
    gt=True

    # Data loading code
    print('root is ', root)
    train_dataset = SemanticKitti(root=root,
                                       sequences=train_sequences,
                                       labels=labels,
                                       color_map=color_map,
                                       learning_map=learning_map,
                                       learning_map_inv=learning_map_inv,
                                       cfg=cfg,
                                       past_frame_skip=2,
                                       max_points=max_points,
                                       gt=gt,
                                       live_compute=live_compute)
    train_dataset.use_rv= True
    train_dataset.use_bev= True
    cover_ratios = []


    if quick_test:
      check_list = np.sort(sample(range(0,train_dataset.__len__()), 3))
    else:
      check_list = range(0, train_dataset.__len__())

    start_t = time.time()
    for index in check_list:
      # print('loading the bev data with index ', index)
      # dp = train_dataset.__get_bev__(index, verbose=is_verbose)
      # print(list(dp.keys()))
      # cover_ratios.append(dp['cover_ratio'])
      if check_sanity:
        pc_label    = dp['new_point_label']
        proj_label  = dp['pixel_cat_map']
        point_coord = dp['new_point_coord']

        miou, acc   = train_dataset.sanity_check(pc_label, proj_label, point_coord, class_ignore=class_ignore, class_strings=labels, class_inv_remap=learning_map_inv, target='bev')
        print('The upper limit is miou: {} acc: {},  discretization under grid_size: {}, extents: {}'.format(miou, acc, train_dataset.voxel_size, train_dataset.area_extents))

      if check_input:
        # p = train_dataset.preprocess_and_save(index, verbose=False)
        p = train_dataset.__getitem__(index, verbose=is_verbose)
        # p = train_dataset.getitem_on_the_fly(index, verbose=is_verbose)
        for j in range(len(p)):
          try:
            print(p[j].shape)
          except:
            pass

    # print('average ratio is ', sum(cover_ratios)/len(cover_ratios))
    print('loading takes {} seconds on average'.format( (time.time() - start_t)/len(check_list)))
    #
