"""
light-weight loader class for contact pts loading
- GT offsets;
- proposal supervision;
-

"""

import numpy as np
import os
import sys
import random
import hydra
import h5py
import torch
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf
from os import makedirs, remove
from os.path import exists, join
import dgl

import __init__
from global_info import global_info
from dataset import obman
from dataset import obman_ho
from dataset.parser import Parser
from dataset.modelnet40 import ModelNetDataset
from common.queries import BaseQueries, TransQueries
from common.debugger import *
from common import vis_utils
from common.train_utils import cycle

epsilon = 10e-8
infos     = global_info()
my_dir    = infos.base_path
second_path = infos.second_path
group_path= infos.group_path
project_path= infos.project_path

DATASETS = []

def add_datasets(module):
  DATASETS.extend([module])

add_datasets(ModelNetDataset)
# add_datasets(HandDatasetComplete)
# add_datasets(HandDatasetAEGan)
# add_datasets(HandDatasetAEGraph)
#
def load_dataset(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  mdict = {dataset.__name__: dataset for dataset in DATASETS}
  if name not in mdict:
    print('Invalid dataset index. Options are:')
    # Display a list of valid dataset names
    for dataset in DATASETS:
      print('\t* {}'.format(dataset.__name__))
    raise ValueError(f'Dataset {name} not defined')
  DatasetClass = mdict[name]

  return DatasetClass

def breakpoint():
    import pdb;pdb.set_trace()

def collate_graph(samples):
    graphs, gt_points, instance_names, Rx, center_offsets, idx, category_name = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    gt_points = torch.stack(gt_points, 0)
    Rx = torch.stack(Rx, 0)
    Tx = torch.stack(center_offsets, 0)
    return {'G': batched_graph, "points": gt_points, "id": instance_names, 'R': Rx, 'T': Tx, 'idx': idx, 'class': category_name}

def collate_graph_partial(samples):
    graphs_raw, graphs_real, n_arr, c_arr, m_arr, gt_points, instance_name, instance_name1, RR, center_offsets, idx, category_name = map(list, zip(*samples))
    batched_graph_raw = dgl.batch(graphs_raw)
    batched_graph_real = dgl.batch(graphs_real)
    gt_points = torch.stack(gt_points, 0)
    n_arr     = torch.stack(n_arr, 0)
    c_arr     = torch.stack(c_arr, 0)
    m_arr     = torch.stack(m_arr, 0)
    Rx        = torch.stack(RR, 0)
    Tx        = torch.stack(center_offsets, 0)
    return {'G': batched_graph_raw, "points": n_arr, "C": c_arr, "id": instance_name, 'R': Rx, 'T':Tx, 'idx': idx, 'class': category_name}

def collate_graph_gan(samples):
    # g, n_arr, gt_points, instance_name, instance_name1
    graphs_raw, graphs_real, n_arr, gt_points, instance_name, instance_name1, RR, idx, category_name = map(list, zip(*samples))
    batched_graph_raw = dgl.batch(graphs_raw)
    batched_graph_real = dgl.batch(graphs_real)
    gt_points = torch.stack(gt_points, 0)
    n_arr     = torch.stack(n_arr, 0)
    Rx        = torch.stack(RR, 0)
    return {'G_raw': batched_graph_raw, 'G_real': batched_graph_real, "raw": n_arr, "real": gt_points, "raw_id": instance_name, "real_id": instance_name1, 'R': Rx, 'idx': idx, 'class': category_name}

def summary(features):
    if isinstance(features, dict):
        for k, v in features.items():
            print(f'type: {k}; Size: {v.size()}')
    else:
        print(f'Size: {features.size()}')
        print(features[0])

class ModelParser(Parser):
    def __init__(self, cfg, mode='train', return_loader=True, domain=None, first_n=-1, add_noise=False, fixed_order=False, num_expr=0.01):
        dset_name  = 'modelnet40'
        collate = None
        if 'Graph' in cfg.dataset_class:
            if cfg.module == 'gan':
                print('---use special collate fn: collate_graph_gan')
                collate = collate_graph_gan
            elif cfg.module == 'ae':
                if 'partial' in cfg.task:
                    print('---use special collate: collate_graph_partial')
                    collate = collate_graph_partial
                else:
                    print('---use special collate fn: collate_graph')
                    collate = collate_graph
        self.train_dataset = ModelNetDataset(cfg=cfg, root=cfg.DATASET.data_path, split='train')
        print("Final train dataset size: {}".format(len(self.train_dataset)))

        self.valid_dataset = ModelNetDataset(cfg=cfg, root=cfg.DATASET.data_path, split='test')
        print("Final valid dataset size: {}".format(len(self.valid_dataset)))

        drop_last = True  # Keeps batch_size constant
        if return_loader:
            self.trainloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=cfg.DATASET.train_batch,
                shuffle=True,
                collate_fn=collate,
                num_workers=int(cfg.DATASET.workers),
                pin_memory=True,
                drop_last=False,
            )
            self.validloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=cfg.DATASET.test_batch,
                shuffle=False,
                collate_fn=collate,
                num_workers=int(cfg.DATASET.workers),
                pin_memory=True,
                drop_last=True,
            )
        else:
            self.validloader = None
            self.trainloader = None
        self.testloader = None

# @hydra.main(config_path="../config/occupancy.yaml")
@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30)
    parser = ModelParser(cfg)
    #
    val_dataset   = parser.valid_dataset
    train_dataset = parser.train_dataset
    val_loader   = parser.validloader
    train_loader   = parser.trainloader

    if cfg.split == 'train':
        dset          = train_dataset
        dloader       = train_loader
    else:
        dset          = val_dataset
        dloader       = val_loader
    j = 0
    g_raw, n_arr, instance_name, up_axis, center_offset, idx= dset.__getitem__(j)
    gt_points = n_arr
    input = g_raw.ndata['x'].numpy()
    gt    = n_arr.transpose(1, 0).numpy()
    full_pts = gt_points.transpose(1, 0).numpy()
    print(f'input: {input.shape}, gt: {gt.shape}')
    vis_utils.plot3d_pts([[input], [gt]], [['input'], ['gt NOCS']],  s=2**2, dpi=300, axis_off=False, color_channel=[[gt], [gt]])
    vis_utils.plot3d_pts([[input], [full_pts]], [['input'], ['full shape']],  s=2**2, dpi=300, axis_off=False)
        # else:
        #     from models.se3net import SE3Transformer
        #     model = SE3Transformer(num_layers=1, atom_feature_size=1, num_degrees=2, num_channels=16, edge_dim=0)
        #     G1, _, _, R1 = dset.__getitem__(0)
        #     out1 = model.forward(G1)
        #
        #     G2, _, _, R2 = dset.__getitem__(10)
        #     out2 = model.forward(G2)
        #     summary(out2)
        #     summary(out1 @ R2)
        #     diff = torch.max(out2 - out1 @ R2).item()
        #     print(diff)


    # for j in range(20):
    #     dp = dset.__getitem__(j)
    #     print('partial: ', dp[-2], 'complete: ', dp[-1])
    #     print(dp[0].shape, dp[1].shape)
    #     vis_utils.visualize_pointcloud([dp[0], dp[1]], title_name='partial + complete', backend='pyrender')
    #     vis_utils.visualize_pointcloud([dp[0]], title_name='partial', backend='pyrender')

    # for j in range(10):
    #     # dset.visualize_original(j)
    #     # dset.visualize_3d_transformed(j)
    #     # dset.visualize_3d_proj(j)

    # # dp = dset.__getitem__(0)
    #     for name, value in dp.items():
    #         try:
    #             print(name, value.shape)
    #         except:
    #             print(name, value)


if __name__ == '__main__':
    main()
# python obman_parser.py training=ae_gan module=vae dataset_class='HandDatasetAEGan' task='adversarial_adaptation' is_testing=True target_category='bottle'
