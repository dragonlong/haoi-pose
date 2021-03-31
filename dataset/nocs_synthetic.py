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

import dgl
import __init__
from global_info import global_info
from common.d3_utils import align_rotation, rotate_about_axis, transform_pcloud

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

def idx_points(points, idx):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class FixedRadiusNearNeighbors(nn.Module):
    '''
    Ball Query - Find the neighbors with-in a fixed radius
    '''
    def __init__(self, radius, n_neighbor, knn=False):
        super(FixedRadiusNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor
        self.knn = knn

    def forward(self, pos, centroids):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        '''
        device = pos.device
        B, N, _ = pos.shape
        # center_pos = idx_points(pos, centroids)
        center_pos = pos
        _, S, _ = center_pos.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        # print('before square_distance ', center_pos.shape, pos.shape)
        sqrdists = square_distance(center_pos, pos)
        if self.knn:
            _, group_idx = torch.topk(sqrdists, self.n_neighbor+1, dim=-1, largest=False, sorted=True)
            group_idx = group_idx[:, :, 1:self.n_neighbor+1]
        else:
            group_idx[sqrdists > self.radius ** 2] = N
            group_idx = group_idx.sort(dim=-1)[0][:, :, :self.n_neighbor]
            group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.n_neighbor])
            mask = group_idx == N
            group_idx[mask] = group_first[mask]
        return group_idx

class NOCSDataset(data.Dataset):
    def __init__(self, cfg, root, split='train', npoint_shift=False, is_testing=False, rand_seed=999):
        self.root  = root
        self.cfg   = cfg
        self.task  = cfg.task
        self.split = split
        # needed number
        self.batch_size    = cfg.DATASET.batch_size
        self.npoints       = cfg.DATASET.num_points
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

        assert(split == 'train' or split == 'val')
        self.target_category = cfg.target_category
        dir_point = os.path.join(self.root, split, str(self.target_category))
        self.datapath = sorted(glob.glob(f'{dir_point}/*/*/*/*.npz'))

        # self.datapath = []
        # for fn in fns:
        #     token = (os.path.splitext(os.path.basename(fn))[0])
        #     self.datapath.append(( cfg.target_category, os.path.join(dir_point, token + '.txt'),
        #                             # os.path.join(dir_point, token + '.qua'),
        #                             # os.path.join(dir_point, token + '.ds'+str(self.num_gen_samples)+'.pt'),
        #                             os.path.join(dir_point, token + '.idx')))
        # create NOCS dict
        manager = Manager()
        self.nocs_dict = manager.dict()
        self.cls_dict = manager.dict()
        self.cloud_dict = manager.dict()
        self.g_dict    = manager.dict()
        self.r_dict    = manager.dict()
        self.frnn  = FixedRadiusNearNeighbors(self.radius, self.num_samples, knn=True)

        if self.cfg.eval or self.split != 'train':
            np.random.seed(0)
            self.random_angle = np.random.rand(self.__len__(), 150, 3) * 360
            self.random_T     = np.random.rand(self.__len__(), 150, 3)

        # pre-fetch
        self.backup_cache = []
        for j in range(100):
            fn  = self.datapath[j]
            category_name = fn.split('.')[0].split('/')[-4]
            instance_name = fn.split('.')[0].split('/')[-5]
            data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
            labels    = data_dict['labels']
            p_arr     = data_dict['points'][labels]
            if p_arr.shape[0] > self.npoints:
                self.backup_cache.append([category_name, instance_name, data_dict, j])

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
            rgb       = data_dict['rgb'][labels]
            pose      = data_dict['pose']
            r, t, s   = pose['rotation'], pose['translation'].reshape(-1, 3), pose['scale']
            labels = labels[labels].astype(np.int).reshape(-1, 1) # only get true
            if s == 0:
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
            # if verbose:
            #     save_path = self.cfg.log_dir + '/input'
            #     if not exists(save_path):
            #         makedirs(save_path)
            #     save_name = save_path + f'/{idx}.txt'
            #     labels    = labels.reshape(-1, 1)
            #     save_arr  = np.concatenate([p_arr, labels], axis=1)
            #     np.savetxt(save_name, save_arr)
            #     save_name = save_path + f'/{idx}_canon.txt'
            #     save_arr  = np.concatenate([n_arr, labels], axis=1)
            #     np.savetxt(save_name, save_arr)
            #     print('saving to ', save_name)
            full_points = np.concatenate([p_arr, n_arr, rgb], axis=1)
            full_points = np.random.permutation(full_points)
            pos         = torch.from_numpy(full_points[:self.npoints, :3].astype(np.float32)).unsqueeze(0)
            nocs_gt     = torch.from_numpy(full_points[:self.npoints, 3:6].astype(np.float32))

            if self.cfg.MODEL.num_in_channels == 1:
                feat= torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))
            else:
                feat      = torch.from_numpy(full_points[:self.npoints, 6:9].astype(np.float32)).unsqueeze(0)
            centroids = torch.from_numpy(np.arange(pos.shape[1]).reshape(1, -1))
            group_idx = self.frnn(pos, centroids)
            src = group_idx[0].contiguous().view(-1)
            dst = centroids[0].view(-1, 1).repeat(1, self.num_samples).view(-1) # real pair

            if self.fetch_cache and idx not in self.g_dict:
                self.g_dict[idx] = [pos, src, dst, feat]

        # construct a graph for input
        unified = torch.cat([src, dst])
        uniq, inv_idx = torch.unique(unified, return_inverse=True)
        src_idx = inv_idx[:src.shape[0]]
        dst_idx = inv_idx[src.shape[0]:]
        g = dgl.DGLGraph((src_idx, dst_idx))
        g.ndata['x'] = pos[0][uniq] # use
        g.ndata['f'] = feat[0][uniq].unsqueeze(-1)
        g.edata['d'] = pos[0][dst_idx] - pos[0][src_idx]

        R = torch.from_numpy(r.astype(np.float32)) # predict r
        T = torch.from_numpy(t.astype(np.float32))
        center = torch.from_numpy(np.array([[0.5, 0.5, 0.5]])) # 1, 3
        center_offset = pos[0].clone().detach() - T #

        return g, nocs_gt, instance_name, R, center_offset, idx, category_name

    def get_sample_full(self, idx, verbose=False):
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
            p_arr     = data_dict['points']
            extra_points = data_dict['points']
            if p_arr.shape[0] < self.npoints:
                category_name, instance_name, data_dict, idx = self.backup_cache[random.randint(0, len(self.backup_cache)-1)]
                labels = data_dict['labels']
                p_arr  = data_dict['points']
            rgb       = data_dict['rgb']
            pose      = data_dict['pose']
            r, t, s   = pose['rotation'], pose['translation'].reshape(-1, 3), pose['scale']
            labels    = labels.astype(np.int).reshape(-1, 1) # only get true
            if s == 0:
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

            full_points = np.concatenate([p_arr, n_arr, rgb, labels], axis=1)
            full_points = np.random.permutation(full_points)
            pos         = torch.from_numpy(full_points[:self.npoints, :3].astype(np.float32)).unsqueeze(0)
            nocs_gt     = torch.from_numpy(full_points[:self.npoints, 3:6].astype(np.float32))
            labels      = torch.from_numpy(full_points[:self.npoints, 9].astype(np.float32)) # N

            if self.cfg.MODEL.num_in_channels == 1:
                feat= torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))
            else:
                feat      = torch.from_numpy(full_points[:self.npoints, 6:9].astype(np.float32)).unsqueeze(0)
            centroids = torch.from_numpy(np.arange(pos.shape[1]).reshape(1, -1))
            group_idx = self.frnn(pos, centroids)
            src = group_idx[0].contiguous().view(-1)
            dst = centroids[0].view(-1, 1).repeat(1, self.num_samples).view(-1) # real pair

            if self.fetch_cache and idx not in self.g_dict:
                self.g_dict[idx] = [pos, src, dst, feat]

        # construct a graph for input
        unified = torch.cat([src, dst])
        uniq, inv_idx = torch.unique(unified, return_inverse=True)
        src_idx = inv_idx[:src.shape[0]]
        dst_idx = inv_idx[src.shape[0]:]
        g = dgl.DGLGraph((src_idx, dst_idx))
        g.ndata['x'] = pos[0][uniq] # use
        g.ndata['f'] = feat[0][uniq].unsqueeze(-1)
        g.edata['d'] = pos[0][dst_idx] - pos[0][src_idx]

        R = torch.from_numpy(r.astype(np.float32)) # predict r
        T = torch.from_numpy(t.astype(np.float32))
        center = torch.from_numpy(np.array([[0.5, 0.5, 0.5]])) # 1, 3
        center_offset = pos[0].clone().detach() - T #

        return g, nocs_gt, instance_name, R, center_offset, idx, category_name, labels

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
    dset = NOCSDataset(cfg=cfg, root='/groups/CESCA-CV/ICML2021/data/nocs/', split='train')
    #
    for i in range(200): #
        dp   = dset.__getitem__(i, verbose=True)
        # print(dp)

if __name__ == '__main__':
    main()
    # python nocs_synthetic.py target_category='5' datasets=nocs_synthetic
