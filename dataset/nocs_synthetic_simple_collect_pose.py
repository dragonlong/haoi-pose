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
import matplotlib.pyplot as plt

def bp():
    import pdb;pdb.set_trace()


class NOCSDataset(data.Dataset):
    def __init__(self, cfg, root, target_category, split='train'):
        self.root = root
        self.cfg  = cfg
        self.split = split
        # needed number
        self.npoints = cfg.num_points
        self.radius = 0.1

        assert(split == 'train' or split == 'val')
        self.target_category = target_category
        dir_point = os.path.join(self.root, split, str(self.target_category))
        self.datapath = sorted(glob.glob(f'{dir_point}/*/*/*/*.npz'))

    def __getitem__(self, idx):
        fn = self.datapath[idx]
        fn_main = fn.split('.')[-2]
        instance_name, idx_0, _, idx_1 = fn_main.split('/')[-4:]
        data_dict = np.load(fn, allow_pickle=True)['all_dict'].item()
        pose = data_dict['pose']
        if pose['scale'] < 0.1 or pose['scale'] > 0.5:
            return None, None
        return instance_name, pose

    def __len__(self):
        return len(self.datapath)


@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False) #

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30)  #
    for target_category in ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']:
        for split in ['train', 'val']:
            dset = NOCSDataset(cfg=cfg, root=cfg.DATASET.data_path,
                               target_category=target_category, split=split)
            instance_pose_list = {}
            for i in range(len(dset)):
                instance, pose = dset[i]
                if instance is None:
                    continue
                if instance not in instance_pose_list:
                    instance_pose_list[instance] = []
                instance_pose_list[instance].append(pose)
            save_path = pjoin(cfg.DATASET.data_path, 'pose_list', f'{target_category}_{split}.npz')
            np.savez_compressed(save_path, data=instance_pose_list)

:memoryview
if __name__ == '__main__':
    main()
    # python nocs_synthetic.py target_category='5' datasets=nocs_synthetic
