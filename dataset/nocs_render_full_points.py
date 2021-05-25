import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from os.path import join as pjoin
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--old_model_pts', type=str)
    parser.add_argument('--category', type=str, default='airplane')
    parser.add_argument('--split', type=str, default='train')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cat2id = {
        'bottle': '02876657',
        'bowl': '02880940',
        'camera': '02942699',
        'can': '02946921',
        'laptop': '03642806',
        'mug': '03797390'
    }
    split = 'val' if args.split == 'test' else args.split
    read_folder = pjoin(args.input, split, cat2id[args.category])
    instance_list = [path.split('/')[-2] for path in [pjoin(read_folder, i, 'model.obj') for i in os.listdir(read_folder)]
                     if os.path.exists(path)]

    output_path = pjoin(args.output, args.category, args.split)
    os.makedirs(output_path, exist_ok=True)

    for instance in instance_list:
        points = np.load(pjoin(args.old_model_pts, f'{instance}.npy'))
        np.savez_compressed(pjoin(output_path, f'{instance}.npz'),
                            points=points)