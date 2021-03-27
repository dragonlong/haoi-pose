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
from torch.utils.data import Dataset
from multiprocessing import Manager
# import open3d as o3d
import joblib

import dgl
import __init__
from global_info import global_info
from common.d3_utils import rotate_eular

def bp():
    import pdb;pdb.set_trace()

infos           = global_info()
my_dir          = infos.base_path
group_path      = infos.group_path
project_path    = infos.project_path
categories_id   = infos.categories_id

def square_distance(src, dst):
    '''
    Adapted from htps://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist #

def idx_points(points, idx):
    '''
    Adapted from htps://github.com/yanx27/Pointnet_Pointnet2_pytorch
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
        Adapted from htps://github.com/yanx27/Pointnet_Pointnet2_pytorch
        '''
        device = pos.device
        B, N, _ = pos.shape
        # center_pos = idx_points(pos, centroids)
        center_pos = pos
        _, S, _ = center_pos.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists  = square_distance(center_pos, pos)
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

class OracleDataset(Dataset):
    def __init__(self, cfg, root, split='train', npoint_shift=False, is_testing=False, rand_seed=999):
        self.root  = root
        self.cfg   = cfg
        self.task  = cfg.task
        self.split = split

        self.npoints       = cfg.num_points
        self.radius        = 0.1
        self.num_samples   = 10

        # augmentation
        self.normalize = cfg.normalize
        self.augment   = cfg.augment
        self.use_partial    = cfg.use_partial
        self.fixed_sampling = cfg.fixed_sampling

        manager    = Manager()
        self.frnn  = FixedRadiusNearNeighbors(self.radius, self.num_samples, knn=True)
        # get data
        self.data_size = 0
        fpath  = f'{my_dir}/data/modelnet40'
        f_train= f'{fpath}/airplane_{split}_2048.pk'
        with open(f_train, "rb") as f:
            self.full_data = joblib.load(f)
        self.data_size = self.full_data.shape[0]

        if self.cfg.eval or self.split != 'train':
            np.random.seed(0)
            self.random_angle = np.random.rand(self.__len__(), 150, 3) * 180
            self.random_T     = np.random.rand(self.__len__(), 150, 3)

    def __len__(self):
        return self.data_size

    def get_sample(self, idx, verbose=False):
        if self.cfg.single_instance:
            idx= 2

        category_name = 'airplane'
        instance_name = f'{idx:04d}'
        raw_pts     = self.full_data[idx]

        boundary_pts = [np.min(raw_pts, axis=0), np.max(raw_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        canon_pts = (raw_pts - center_pt.reshape(1, 3))/length_bb
        # if self.use_partial:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points  = o3d.utility.Vector3dVector(raw_pts)
        #     pcd.normals = o3d.utility.Vector3dVector(point_normal_set[:, 3:])
        #     camera_location, radius = np.matmul(np.array([[2, 2, 2]]), r.T), 1000 # from randomly rotate
        #     camera_location = camera_location.astype(np.float64).reshape(3, 1)
        #     visible_pts = pcd.hidden_point_removal(camera_location, radius)
        #     canon_pts = canon_pts[visible_pts[1]]
        if self.fixed_sampling:
            pos = torch.from_numpy(np.copy(canon_pts)[:self.npoints, :]).unsqueeze(0)
        else:
            pos = torch.from_numpy(np.random.permutation(np.copy(canon_pts))[:self.npoints, :]).unsqueeze(0)

        gt_points = pos[0].clone().detach()
        # canonical
        center_offset = pos[0].clone().detach() # to 0
        center = torch.from_numpy(np.array([[0.0, 0.0, 0.0]])) # 1, 3
        up_axis= torch.tensor([[0.0, 1.0, 0.0]]).float() # y
        if self.cfg.eval:
            theta_x = self.random_angle[idx, self.cfg.iteration, 0]
            theta_z = self.random_angle[idx, self.cfg.iteration, 2]
            theta_y = 0
            r = rotate_eular(theta_x, theta_y, theta_z)
            t = self.random_T[idx, self.cfg.iteration].reshape(1, 3).astype(np.float32)
        elif self.augment:
            theta_x = random.randint(0, 180)
            theta_z = random.randint(0, 180)
            theta_y = 0
            r = rotate_eular(theta_x, theta_y, theta_z)
            t = np.random.rand(1,3).astype(np.float32)
        else:
            r = np.eye(3).astype(np.float32)#
            t = np.array([[0, 0, 0]]).astype(np.float32)

        R     = torch.from_numpy(r.astype(np.float32))
        T     = torch.from_numpy(t.astype(np.float32))

        pos[0] = torch.matmul(pos[0], R.T) # to have random rotation
        pos[0] = pos[0] + T
        center_offset = pos[0] - T  # N, 3
        up_axis = torch.matmul(up_axis, R.T)
        nocs_pts= gt_points + 0.5 #
        g = self.__build_graph__(pos, k=self.num_samples)
        return g, nocs_pts, instance_name, R, center_offset, idx, category_name

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        return sample

    def __build_graph__(self, pos, k=10):
        centroids = torch.from_numpy(np.arange(pos.shape[1]).reshape(1, -1))
        group_idx = self.frnn(pos, centroids)

        src = group_idx[0].contiguous().view(-1)
        dst = centroids[0].view(-1, 1).repeat(1, k).view(-1) # real pair
        feat= torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32)) # if there is no feature

        # construct a graph for input
        unified = torch.cat([src, dst])
        uniq, inv_idx = torch.unique(unified, return_inverse=True)
        src_idx = inv_idx[:src.shape[0]]
        dst_idx = inv_idx[src.shape[0]:]

        g = dgl.DGLGraph((src_idx, dst_idx))
        g.ndata['x'] = pos[0][uniq]
        g.ndata['f'] = feat[0][uniq].unsqueeze(-1)
        g.edata['d'] = pos[0][dst_idx] - pos[0][src_idx] #[num_atoms,3] but we only supervise the half

        return g

@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getatr and hasatr methods to function correctly

    #>>>>>>>>>>>>>>>>> seting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30)
    dset = OracleDataset(cfg=cfg, root='/home/dragon/Documents/ICML2021/data/modelnet40/', split='train')
    dp   = dset.__getitem__(0)
    print(dp)

# python modelnet40.py datasets=modelnet40
if __name__ == '__main__':

    main()
