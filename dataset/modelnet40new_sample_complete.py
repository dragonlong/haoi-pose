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


def create_complete(read_path, save_folder, ins_num, num_points=10000, vis=False):
    m = trimesh.load(read_path)
    # centralization
    c = np.mean(m.vertices, axis=0)
    trans = np.eye(4)
    trans[:3, 3] = -c
    m.apply_transform(trans)
    scale = np.max(np.sqrt(np.sum(m.vertices ** 2, axis=1)))
    trans = np.eye(4)
    trans[:3, :3] = np.eye(3) / scale
    m.apply_transform(trans)

    pts, face_index = trimesh.sample.sample_surface(m, num_points)

    np.savez_compressed(pjoin(save_folder, f'{ins_num}.npz'), points=pts)

    if vis:
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        center = (pmin + pmax) * 0.5
        lim = max(pmax - pmin) * 0.5 + 0.2

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=1)
        ax.set_xlim3d([center[0] - lim, center[0] + lim])
        ax.set_ylim3d([center[1] - lim, center[1] + lim])
        ax.set_zlim3d([center[2] - lim, center[2] + lim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


def create_complete_bottle_for_nocs(read_path, save_folder, ins_num, num_points=10000, vis=False):
    m = trimesh.load(read_path)
    trans = np.eye(4)
    trans[:3, :3] = np.asarray([[0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0]])
    m.apply_transform(trans)
    bbox_up, bbox_down = np.max(m.vertices, axis=0, keepdims=True), np.min(m.vertices, axis=0, keepdims=True)
    center = 0.5 * (bbox_up + bbox_down)
    scale = np.sqrt(np.sum((bbox_up - bbox_down) ** 2, axis=1))
    trans = np.eye(4)
    trans[:3, 3] = -center
    m.apply_transform(trans)
    trans = np.eye(4)
    trans[:3, :3] = np.eye(3) / scale
    m.apply_transform(trans)

    pts, face_index = trimesh.sample.sample_surface(m, num_points)

    np.savez_compressed(pjoin(save_folder, f'{ins_num}.npz'), points=pts)

    if vis:
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        center = (pmin + pmax) * 0.5
        lim = max(pmax - pmin) * 0.5 + 0.2

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=1)
        ax.set_xlim3d([center[0] - lim, center[0] + lim])
        ax.set_ylim3d([center[1] - lim, center[1] + lim])
        ax.set_zlim3d([center[2] - lim, center[2] + lim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


def proc_render(path_list, save_folder, num_points=10000, for_nocs=False):
    for read_path in tqdm(path_list):
        ins_num = read_path.split('/')[-1].split('.')[-2].split('_')[-1]
        if for_nocs:
            create_complete_bottle_for_nocs(read_path, save_folder, ins_num, num_points=num_points)
        else:
            create_complete(read_path, save_folder, ins_num, num_points=num_points)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--category', type=str, default='airplane')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_proc', type=int, default=8)
    parser.add_argument('--for_nocs', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args() 
    
    read_folder = pjoin(args.input, args.category, args.split)
    save_folder = pjoin(args.output, args.category, args.split)
    path_list = [pjoin(read_folder, i) for i in os.listdir(read_folder) if i.endswith('off')]

    os.makedirs(save_folder, exist_ok=True)

    mp.set_start_method('spawn')
    num_per_ins = (len(path_list) - 1) // args.num_proc + 1
    processes = []
    for i in range(args.num_proc):
        st = num_per_ins * i
        ed = min(st + num_per_ins, len(path_list))
        p = mp.Process(target=proc_render, args=(path_list[st: ed], save_folder, args.num_points,
                                                 args.for_nocs))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()

