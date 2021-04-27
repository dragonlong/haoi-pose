#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import random
import hydra
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf
from os import makedirs, remove
from os.path import exists, join
import glob
import os.path
import json
import h5py
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms
from multiprocessing import Manager

import scipy.io as sio
from scipy.spatial.transform import Rotation as sciR

import __init__
from global_info import global_info
from common.d3_utils import align_rotation, rotate_about_axis, transform_pcloud
import vgtk.pc as pctk
import vgtk.point3d as p3dtk
import vgtk.so3conv.functional as L
from vgtk.functional import rotation_distance_np, label_relative_rotation_np

def bp():
    import pdb;pdb.set_trace()

infos           = global_info()
my_dir          = infos.base_path
group_path      = infos.group_path
project_path    = infos.project_path
categories_id   = infos.categories_id

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist #

class NOCSDataset(data.Dataset):
    def __init__(self, cfg, split='train', root=None, npoint_shift=False, is_testing=False, rand_seed=999):
        if root is None:
            self.root = cfg.DATASET.data_path
        else:
            self.root  = root
        self.cfg   = cfg
        self.task  = cfg.task
        self.split = split
        # needed number
        self.npoints       = cfg.num_points
        self.num_gen_samples=cfg.DATASET.num_gen_samples
        self.num_of_class  =cfg.DATASET.num_of_class
        self.radius = 0.1
        self.num_samples = 10 # default to be 10

        # augmentation
        self.normalize = cfg.normalize
        self.augment   = cfg.augment

        if self.task == 'category_pose':
            self.is_gen       = cfg.is_gen
            self.is_debug     = cfg.is_debug
            self.nocs_type    = cfg.nocs_type

        self.is_testing   = is_testing

        self.fetch_cache  = cfg.fetch_cache
        shape_ids = {}

        # attention method: 'attention | rotation'

        self.anchors = L.get_anchors()

        assert(split == 'train' or split == 'val')
        self.target_category = cfg.target_category
        dir_point = os.path.join(self.root, split, str(self.target_category))
        self.datapath = sorted(glob.glob(f'{dir_point}/*/*/*/*.npz'))

        # create NOCS dict
        manager = Manager()
        self.nocs_dict = manager.dict()
        self.cls_dict = manager.dict()
        self.cloud_dict = manager.dict()
        self.g_dict    = manager.dict()
        self.r_dict    = manager.dict()

        if self.cfg.eval or self.split != 'train':
            np.random.seed(0)
            self.random_angle = np.random.rand(self.__len__(), 150, 3) * 360
            self.random_T     = np.random.rand(self.__len__(), 150, 3)

        # pre-fetch
        self.backup_cache = []
        for j in range(300):
            fn  = self.datapath[j]
            category_name = fn.split('.')[0].split('/')[-4]
            instance_name = fn.split('.')[0].split('/')[-5]
            data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
            labels    = data_dict['labels']
            p_arr     = data_dict['points'][labels]
            if p_arr.shape[0] > self.npoints:
                self.backup_cache.append([category_name, instance_name, data_dict, j])
        print('backup has ', len(self.backup_cache))

    def get_sample_partial(self, idx, verbose=False):
        fn  = self.datapath[idx]
        if verbose:
            print(fn)
        category_name = fn.split('.')[0].split('/')[-4]
        instance_name = fn.split('.')[0].split('/')[-5]

        if self.fetch_cache and idx in self.g_dict:
            pos, src, dst, feat = self.g_dict[idx]
        else:
            data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
            labels    = data_dict['labels']
            p_arr  = data_dict['points'][labels]
            if p_arr.shape[0] < self.npoints:
                category_name, instance_name, data_dict, idx = self.backup_cache[random.randint(0, len(self.backup_cache)-1)]
                if verbose:
                    print('use ', idx)
                labels = data_dict['labels']
                p_arr  = data_dict['points'][labels]
            rgb       = data_dict['rgb'][labels] / 255.0
            pose      = data_dict['pose']
            r, t, s   = pose['rotation'], pose['translation'].reshape(-1, 3), pose['scale']
            labels = labels[labels].astype(np.int).reshape(-1, 1) # only get true
            if s < 0.00001:
                s = 1
            n_arr     = np.matmul(p_arr - t, r) / s + 0.5 # scale
            center    = t.reshape(1, 3)
            bb_pts    = np.array([[0.5, 0.5, 0.5]])
            bb_pts    = s * np.matmul(bb_pts, r.T)  + t.reshape(1, 3) # we actually don't know the exact bb, sad
            center_offset = p_arr - center #
            bb_offset =  bb_pts - p_arr #
            up_axis   = np.matmul(np.array([[0.0, 1.0, 0.0]]), r.T)
            if verbose:
                print(f'we have {p_arr.shape[0]} pts')
            full_points = np.concatenate([p_arr, n_arr, rgb], axis=1)
            full_points = np.random.permutation(full_points)
            pos         = torch.from_numpy(full_points[:self.npoints, :3].astype(np.float32)).unsqueeze(0)
            nocs_gt     = torch.from_numpy(full_points[:self.npoints, 3:6].astype(np.float32))

            if self.cfg.MODEL.num_in_channels == 1:
                feat = torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))
            elif self.cfg.use_rgb:
                feat = torch.from_numpy(full_points[:self.npoints, 6:9].astype(np.float32)).unsqueeze(0)
            else:
                feat = torch.from_numpy(full_points[:self.npoints, 6:9].astype(np.float32)).unsqueeze(0)

        T = torch.from_numpy(t.astype(np.float32))
        _, R_label, R0 = rotation_distance_np(r, self.anchors)
        R_gt = torch.from_numpy(r.astype(np.float32)) # predict r
        center = torch.from_numpy(np.array([[0.5, 0.5, 0.5]])) # 1, 3
        center_offset = pos[0].clone().detach() - T #

        return {'pc': pos[0]/s, # normalize
                'label': torch.from_numpy(np.array([1])).long(),
                'R': R0,
                'id': idx,
                'R_gt' : R_gt,
                'R_label': torch.Tensor([R_label]).long(),
               }

    def get_sample_full(self, idx, verbose=False):
        fn  = self.datapath[idx]
        category_name = fn.split('.')[0].split('/')[-4]
        instance_name = fn.split('.')[0].split('/')[-5]

        if self.fetch_cache and idx in self.g_dict:
            pos, src, dst, feat = self.g_dict[idx]
        else:
            data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
            labels    = data_dict['labels']
            p_arr     = data_dict['points']
            if p_arr.shape[0] < self.npoints:
                category_name, instance_name, data_dict, idx = self.backup_cache[random.randint(0, len(self.backup_cache)-1)]
                labels = data_dict['labels']
                p_arr  = data_dict['points']
            ind_fore  = np.where(labels)[0]
            ind_back  = np.where(~labels)[0]

            rgb       = data_dict['rgb'] / 255.0
            pose      = data_dict['pose']
            r, t, s   = pose['rotation'], pose['translation'].reshape(-1, 3), pose['scale']
            labels    = labels.astype(np.int)
            if s < 0.00001:
                s = 1
            n_arr     = np.matmul(p_arr - t, r) / s + 0.5 # scale
            center    = t.reshape(1, 3)
            bb_pts    = np.array([[0.5, 0.5, 0.5]])
            bb_pts    = s * np.matmul(bb_pts, r.T)  + t.reshape(1, 3) # we actually don't know the exact bb, sad
            center_offset = p_arr - center #
            bb_offset =  bb_pts - p_arr #
            up_axis   = np.matmul(np.array([[0.0, 1.0, 0.0]]), r.T)

            # choose inds
            half_num    = int(self.npoints/2)
            if len(ind_back) < half_num:
                half_num = self.npoints - len(ind_back)
            fore_choose = np.random.permutation(ind_fore)[:half_num]
            another_half_num = self.npoints - len(fore_choose)
            back_choose = np.random.permutation(ind_back)[:another_half_num]
            all_inds    = np.concatenate([fore_choose, back_choose])
            pos         = torch.from_numpy(p_arr[all_inds].astype(np.float32)).unsqueeze(0)
            nocs_gt     = torch.from_numpy(n_arr[all_inds].astype(np.float32))
            labels      = torch.from_numpy(labels[all_inds].astype(np.float32)) # N

            if self.cfg.MODEL.num_in_channels == 1:
                feat= torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))
            else:
                feat      = torch.from_numpy(rgb[all_inds].astype(np.float32)).unsqueeze(0)

        T = torch.from_numpy(t.astype(np.float32))
        _, R_label, R0 = rotation_distance_np(r, self.anchors)
        R_gt = torch.from_numpy(r.astype(np.float32)) # predict r
        center = torch.from_numpy(np.array([[0.5, 0.5, 0.5]])) # 1, 3
        center_offset = pos[0].clone().detach() - T #

        return {'pc': pos[0]/s,
                'label': torch.from_numpy(np.array([1])).long(),
                'R': R0,
                'id': idx,
                'R_gt' : R_gt,
                'R_label': torch.Tensor([R_label]).long(),
               }

    def __getitem__(self, idx, verbose=False):
        if self.cfg.use_background:
            sample = self.get_sample_full(idx, verbose=verbose)
        else:
            sample = self.get_sample_partial(idx, verbose=verbose)
        return sample

    def __len__(self):
        return len(self.datapath)

@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False) #

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30) #
    cfg.item     ='nocs_synthetic'
    cfg.name_dset='nocs_synthetic'
    cfg.log_dir  = infos.second_path + cfg.log_dir
    dset = NOCSDataset(cfg=cfg, split='train')
    #
    for i in range(200): #
        dp   = dset.__getitem__(i, verbose=True)
        # print(dp)

if __name__ == '__main__':
    main()
    # python nocs_synthetic.py target_category='5' datasets=nocs_synthetic
