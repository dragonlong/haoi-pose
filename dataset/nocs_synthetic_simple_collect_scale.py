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

from os.path import join as pjoin
base_path = os.path.dirname(__file__)
sys.path.insert(0, pjoin(base_path, '..'))
import __init__
from global_info import global_info
import matplotlib.pyplot as plt

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
    def __init__(self, cfg, root, target_category, split='train'):
        self.root = root
        self.cfg  = cfg
        self.split = split
        # needed number
        self.npoints = cfg.num_points
        self.num_gen_samples=cfg.DATASET.num_gen_samples
        self.num_of_class  =cfg.DATASET.num_of_class
        self.radius = 0.1
        self.num_samples = 10 # default to be 10

        # augmentation
        self.normalize = cfg.normalize
        self.augment   = cfg.augment

        assert(split == 'train' or split == 'val')
        self.target_category = target_category
        dir_point = os.path.join(self.root, split, str(self.target_category))
        self.datapath = sorted(glob.glob(f'{dir_point}/*/*/*/*.npz'))

    def __getitem__(self, idx):
        fn = self.datapath[idx]
        data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
        pose = data_dict['pose']
        r, t, s = pose['rotation'], pose['translation'].reshape(-1, 3), pose['scale']
        return s

    def __len__(self):
        return len(self.datapath)

@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False) #

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30)  #
    cfg.item = 'nocs_synthetic'
    cfg.name_dset = 'nocs_synthetic'
    cfg.log_dir = infos.second_path + cfg.log_dir
    for target_category in ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']:
        all_scale = []
        for split in ['train', 'val']:
            dset = NOCSDataset(cfg=cfg, root=cfg.DATASET.data_path,
                               target_category=target_category, split=split)
            for i in range(len(dset)):
                s = dset[i]
                all_scale.append(s)
        all_scale = np.asarray(all_scale)
        all_scale.sort()
        print(f'------{target_category}---------')
        print('smallest 5:', all_scale[:5])
        print('largest 5:', all_scale[-5:])


if __name__ == '__main__':
    main()
    # python nocs_synthetic.py target_category='5' datasets=nocs_synthetic
