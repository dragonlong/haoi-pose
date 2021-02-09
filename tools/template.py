import os
import glob
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import hydra
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch

# custom
import __init__
from global_info import global_info
from dataset.obman_parser import ObmanParser
from common.train_utils import CheckpointIO
from common import bp

# try using custom packages
infos     = global_info()
my_dir    = infos.base_path
second_path = infos.second_path
render_path = infos.render_path
viz_path  = infos.viz_path
hand_mesh = infos.hand_mesh
hand_urdf = infos.hand_urdf
grasps_meta  = infos.grasps_meta
mano_path    = infos.mano_path

whole_obj = infos.whole_obj
part_obj  = infos.part_obj
obj_urdf  = infos.obj_urdf

def get_pairs():
    # fetch all needed
    A, B   = 1, 2
    A1, B1 = 1, 2

    return [[A, B], [A1, B1]]

def get_index_per_category():
    #
    all_ids = [[], []]
    return all_ids
    # data path

    # prediction path

@hydra.main(config_path="../config/occupancy.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    # Set t0
    t0 = time.time()
    # category-wise training setup
    infos       = global_info()
    data_infos  = infos.datasets[cfg.item]

    cfg.root_data   = infos.second_path + '/data'
    cfg.log_dir     = infos.second_path + cfg.log_dir
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    # Shorthands
    out_dir    = cfg.log_dir
    print('Saving to ', out_dir)

if __name__ == '__main__':
    main()
