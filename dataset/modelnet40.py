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

class ModelNetDataset(data.Dataset):
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

        self.catfile = os.path.join(self.root, 'modelnet'+str(self.num_of_class)+'_shape_names.txt')
        self.cat     = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        if self.task == 'category_pose':
            self.is_gen       = cfg.is_gen
            self.is_debug     = cfg.is_debug
            self.nocs_type    = cfg.nocs_type

        self.is_testing   = is_testing

        self.fetch_cache  = cfg.fetch_cache
        shape_ids = {}

        shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet'+str(self.num_of_class)+'_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet'+str(self.num_of_class)+'_test.txt'))]

        assert(split == 'train' or split == 'test')

        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]

        # list of (shape_name, shape_txt_file_path) tuple
        self.target_category = cfg.target_category
        if self.target_category == '':
            self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.txt',
                                              # os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.qua',
                                              # os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.ds'+str(self.num_gen_samples)+'.pt',
                                              os.path.join(self.root, shape_names[i],  shape_ids[split][i])+'.idx') for i in range(len(shape_ids[split]))]

        else:
            dir_point = os.path.join(self.root, self.target_category)
            fns=[]
            for file in os.listdir(dir_point):
                if file.endswith(".txt"):
                    fns.append(file)
            fns = sorted(fns)

            if split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in shape_ids['train']]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in shape_ids['test']]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            self.datapath = []
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.datapath.append(( cfg.target_category, os.path.join(dir_point, token + '.txt'),
                                        # os.path.join(dir_point, token + '.qua'),
                                        # os.path.join(dir_point, token + '.ds'+str(self.num_gen_samples)+'.pt'),
                                        os.path.join(dir_point, token + '.idx')))

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

    def get_sample_complete(self, idx, verbose=False):
        if self.cfg.eval or self.split != 'train':
            theta_x = self.random_angle[idx, self.cfg.iteration, 0]
            Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
            theta_z = self.random_angle[idx, self.cfg.iteration, 2]
            Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
            RR = torch.from_numpy(np.matmul(Rx, Rz).astype(np.float32))
            TT = torch.from_numpy(self.random_T[idx, self.cfg.iteration].reshape(1, 3).astype(np.float32))
        elif self.augment and idx!=0:
            theta_x = random.randint(0, 180)
            Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
            theta_z = random.randint(0, 180)
            Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
            RR = torch.from_numpy(np.matmul(Rx, Rz).astype(np.float32))
            TT = torch.from_numpy(np.random.rand(1,3).astype(np.float32))
        else:
            RR = torch.from_numpy(rotate_about_axis(0, axis='x').astype(np.float32))
            TT = 0
        if self.cfg.single_instance:
            idx= 2

        # category_name = model_path.split('/')[-4]
        # instance_name = model_path.split('/')[-3]
        fn  = self.datapath[idx]
        cls = np.array([self.classes[fn[0]]]).astype(np.int32)
        category_name = fn[0]
        instance_name = fn[1].split('/')[-1].split('.')[0].split('_')[1]

        if self.fetch_cache and idx in self.g_dict:
            gt_points, src, dst = self.g_dict[idx]
            pos = gt_points.clone().detach().unsqueeze(0)
            feat= torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))
        else:
            point_normal_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            canon_pts=point_normal_set[:,0:3]
            #
            boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
            center_pt = (boundary_pts[0] + boundary_pts[1])/2
            length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
            # best,
            gt_points = (canon_pts - center_pt.reshape(1, 3)) / length_bb + 0.5
            gt_points = np.random.permutation(gt_points)
            gt_points = gt_points[:self.npoints, :]
            gt_points = torch.from_numpy(gt_points.astype(np.float32)[:, :])

            pos = gt_points.clone().detach().unsqueeze(0)
            centroids = torch.from_numpy(np.arange(gt_points.shape[0]).reshape(1, -1))
            group_idx = self.frnn(pos, centroids)

            src = group_idx[0].contiguous().view(-1)
            dst = centroids[0].view(-1, 1).repeat(1, self.num_samples).view(-1) # real pair
            feat      = torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))

            if self.fetch_cache and idx not in self.g_dict:
                self.g_dict[idx] = [gt_points, src, dst]

        center_offset = pos[0].clone().detach()-0.5
        if self.augment:
            pos[0] = torch.matmul(pos[0]-0.5, RR) + 0.5 # to have random rotation
            pos[0] = pos[0] + TT
            center_offset = torch.matmul(center_offset, RR) # N, 3

        # construct a graph for input
        unified = torch.cat([src, dst])
        uniq, inv_idx = torch.unique(unified, return_inverse=True)
        src_idx = inv_idx[:src.shape[0]]
        dst_idx = inv_idx[src.shape[0]:]
        if verbose:
            print('src_idx.shape', src_idx.shape, '\n', src_idx[0:100], '\n', 'dst_idx.shape', dst_idx.shape, '\n', dst_idx[0:100])
        g = dgl.DGLGraph((src_idx, dst_idx))
        g.ndata['x'] = pos[0][uniq]
        g.ndata['f'] = feat[0][uniq].unsqueeze(-1)
        g.edata['d'] = pos[0][dst_idx] - pos[0][src_idx] #[num_atoms,3] but we only supervise the half
        up_axis = torch.matmul(torch.tensor([[0.0, 1.0, 0.0]]).float(), RR)

        if self.cfg.pred_6d:
            return g, gt_points.transpose(1, 0), instance_name, RR, center_offset, idx, category_name
        else:
            return g, gt_points.transpose(1, 0), instance_name, up_axis, center_offset, idx, category_name

    def __getitem__(self, idx):
        # if idx in self.cache:
        #     point_normal_set, lrf_set, ds_idx_set, wrong_ids, cls = self.cache[idx]
        sample = self.get_sample_complete(idx)

        return sample

    def __len__(self):
        return len(self.datapath)

@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30)
    dset = ModelNetDataset(cfg=cfg, root=cfg.DATASET.data_path, split='train')
    dp   = dset.__getitem__(0)
    print(dp)
    # parser = ObmanParser(cfg)
    # val_dataset   = parser.valid_dataset
    # train_dataset = parser.train_dataset
    # val_loader   = parser.validloader
    # train_loader   = parser.trainloader

if __name__ == '__main__':

    main()
