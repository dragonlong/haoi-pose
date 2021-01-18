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

import __init__
from global_info import global_info
from dataset import obman
from dataset import obman_ho
from dataset.obman_handataset import HandDataset
from dataset.parser import Parser
from common.queries import BaseQueries, TransQueries

epsilon = 10e-8
infos     = global_info()
my_dir    = infos.base_path
second_path = infos.second_path
group_path= infos.group_path
project_path= infos.project_path

def breakpoint():
    import pdb;pdb.set_trace()

def get_dataset(cfg,
                dset_name,
                split,
                train_it=True,
                black_padding=False,
                max_queries=None,
                use_cache=True):
    # begin here
    task=cfg.task
    meta={
        "mode": cfg.DATASET.mode,
        "override_scale": cfg.DATASET.override_scale,
        "fhbhands_split_type": cfg.DATASET.fhbhands_split_type,
        "fhbhands_split_choice": cfg.DATASET.fhbhands_split_choice,
        "fhbhands_topology": cfg.DATASET.fhbhands_topology,
    }
    max_queries=max_queries
    point_nb=cfg.DATASET.atlas_points_nb
    center_idx=cfg.DATASET.center_idx
    limit_size=cfg.limit_size
    sides=cfg.DATASET.sides
    if cfg.use_hand_occupancy:
        pose_dataset = obman_ho.ObMan_HO(
            split=split,
            use_cache=use_cache,
            mode=meta["mode"],
            mini_factor=cfg.DATASET.mini_factor,
            override_scale=meta["override_scale"],
            segment=False,
            use_external_points=True,
            shapenet_root=f"{group_path}/external/ShapeNetCore.v2",
            obman_root=f"{group_path}/external/obman/obman")
    else:
        pose_dataset = obman.ObMan(
            split=split,
            use_cache=use_cache,
            mode=meta["mode"],
            mini_factor=cfg.DATASET.mini_factor,
            override_scale=meta["override_scale"],
            segment=False,
            use_external_points=True,
            shapenet_root=f"{group_path}/external/ShapeNetCore.v2",
            obman_root=f"{group_path}/external/obman/obman")

    # Find maximal dataset-compatible queries
    queries = set(max_queries).intersection(set(pose_dataset.all_queries))
    max_rot = np.pi
    scale_jittering  = 0.3
    center_jittering = 0.2

    if "override_scale" not in meta:
        meta["override_scale"] = False

    dataset = HandDataset(
        pose_dataset,
        cfg=cfg,
        black_padding=black_padding,
        block_rot=False,
        sides=sides,
        train=train_it,
        max_rot=max_rot,
        normalize_img=False,
        center_idx=center_idx,
        point_nb=point_nb,
        scale_jittering=scale_jittering,
        center_jittering=center_jittering,
        queries=queries,
        as_obj_only=meta["override_scale"],
    )
    if limit_size is not None:
        if len(dataset) < limit_size:
            warnings.warn(
                "limit size {} < dataset size {}, working with full dataset".format(
                    limit_size, len(dataset)
                )
            )
        else:
            warnings.warn(
                "Working wth subset of {} of size {}".format(dset_name, limit_size)
            )
            dataset = Subset(dataset, list(range(limit_size)))
    return dataset

def get_queries(cfg):
    max_queries = [
        BaseQueries.occupancy,
        BaseQueries.depth,
        BaseQueries.pcloud,
        BaseQueries.nocs,
        TransQueries.nocs,
    ]
    # max_queries = [
    #     TransQueries.affinetrans,
    #     TransQueries.images,
    #     TransQueries.verts3d,
    #     TransQueries.center3d,
    #     TransQueries.joints3d,
    #     TransQueries.objpoints3d,
    #     TransQueries.pcloud,
    #     TransQueries.camintrs,
    #     BaseQueries.sides,
    #     BaseQueries.camintrs,
    #     BaseQueries.meta,
    #     BaseQueries.segms,
    #     BaseQueries.depth,
    #     TransQueries.sdf,
    #     TransQueries.sdf_points,
    # ]
    if cfg.DATASET.mano_lambda_joints2d:
        max_queries.append(TransQueries.joints2d)
    return max_queries

class ObmanParser(Parser):
    def __init__(self, cfg, mode='train', return_loader=True, domain=None, first_n=-1, add_noise=False, fixed_order=False, num_expr=0.01):
        limit_size = cfg.limit_size
        dset_name  = 'obman'
        max_queries= get_queries(cfg)

        if mode != 'debug':
            self.train_dataset = get_dataset(
                cfg,
                dset_name,
                max_queries=max_queries,
                split='train',
                train_it=True)
            print("Final dataset size: {}".format(len(self.train_dataset)))

            if return_loader:
                # Initialize train dataloader
                self.trainloader = torch.utils.data.DataLoader(
                    self.train_dataset,
                    batch_size=cfg.DATASET.train_batch,
                    shuffle=True,
                    num_workers=int(cfg.DATASET.workers / len(cfg.DATASET.train_splits)),
                    pin_memory=True,
                    drop_last=True,
                )
            else:
                self.trainloader = None
        else:
            self.trainloader = None

        self.valid_dataset = get_dataset(cfg,
            dset_name,
            max_queries=max_queries,
            split='val',
            train_it=False)

        drop_last = True  # Keeps batch_size constant
        if return_loader:
            self.validloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=cfg.DATASET.test_batch,
                shuffle=False,
                num_workers=int(cfg.DATASET.workers / len(cfg.DATASET.val_datasets)),
                pin_memory=True,
                drop_last=drop_last,
            )
        else:
            self.validloader = None
        self.testloader = None

@hydra.main(config_path="../config/occupancy.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30)
    parser = ObmanParser(cfg, mode='debug')
    dset   = parser.valid_dataset
    # dp = dset.__getitem__(0)
    for j in range(10):
        # dset.visualize_original(j)
        dset.visualize_3d_transformed(j)
        # dset.visualize_3d_proj(j)
    # dp = dset.__getitem__(0)
    # for name, value in dp.items():
    #     try:
    #         print(name, value.shape)
    #     except:
    #         print(name, value)


if __name__ == '__main__':
    main()
