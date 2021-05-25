import trimesh
import numpy as np
import os
import multiprocessing as mp
import gc
import argparse
from tqdm import tqdm
import pyrender
import matplotlib.pyplot as plt
import cv2
from os.path import join as pjoin
import pickle

import sys
cur_path = os.path.dirname(__file__)
sys.path.insert(0, pjoin(cur_path, '..', '..'))

from common.vis_utils import plot_distribution
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_input', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    for category in ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']:
        for split in ['train', 'val']:
            pose_dict = np.load(pjoin(args.pose_input, f'{category}_{split}.npz'), allow_pickle=True)['data'].item()
            all_s = []
            for instance in pose_dict:
                for pose in pose_dict[instance]:
                    all_s.append(pose['scale'])

            all_s = np.asarray(all_s)
            all_s.sort()

            plot_distribution(all_s, labelx='scale', labely='frequency', title_name=f'{category}_{split}_scale',
                              sub_name='', save_fig=True)


