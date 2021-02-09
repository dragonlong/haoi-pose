import random
import traceback
import csv

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image, ImageFilter
import pickle
from tqdm import tqdm
import os
from os import makedirs, remove
from os.path import exists, join
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms
import __init__
from global_info import global_info
from dataset.obman_handataset import HandDataset
from common import data_utils, handutils, vis_utils, bp
from common.d3_utils import align_rotation #4 * 4
from common.queries import (
    BaseQueries,
    TransQueries,
    one_query_in,
    no_query_in,
)
def bp():
    import pdb;pdb.set_trace()

infos           = global_info()
my_dir          = infos.base_path
project_path    = infos.project_path
categories_id   = infos.categories_id

def bbox_from_joints(joints):
    x_min, y_min = joints.min(0)
    x_max, y_max = joints.max(0)
    bbox = [x_min, y_min, x_max, y_max]
    return bbox

class HandDatasetComplete(HandDataset):
    """Class inherited by hands datasets
    hands datasets must implement the following methods:
    - get_image
    that respectively return a PIL image and a numpy array
    - the __len__ method

    and expose the following attributes:
    - the cache_folder : the path to the cache folder of the dataset
    """
    def __init__(
        self,
        pose_dataset,
        cfg=None,
        center_idx=9,
        point_nb=600,
        inp_res=256,
        max_rot=np.pi,
        normalize_img=False,
        split="train",
        scale_jittering=0.3,
        center_jittering=0.2,
        train=True,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
        queries=[
            BaseQueries.images,
            TransQueries.joints2d,
            TransQueries.verts3d,
            TransQueries.joints3d,
            TransQueries.depth,
            TransQueries.sdf,
            TransQueries.sdf_points,
        ],
        sides="both",
        block_rot=False,
        black_padding=False,
        as_obj_only=False,
        is_testing=False
    ):
        """
        Args:
        center_idx: idx of joint on which to center 3d pose
        as_obj_only: apply same centering and scaling as when objects are
            not present
        sides: if both, don't flip hands, if 'right' flip all left hands to
            right hands, if 'left', do the opposite
        """
        HandDataset.__init__(
                self,
                pose_dataset=pose_dataset,
                cfg=cfg,
                center_idx=center_idx,
                point_nb=point_nb,
                inp_res=inp_res,
                max_rot=max_rot,
                normalize_img=normalize_img,
                split=split,
                scale_jittering=scale_jittering,
                center_jittering=center_jittering,
                train=train,
                hue=hue,
                saturation=saturation,
                contrast=contrast,
                brightness=brightness,
                blur_radius=blur_radius,
                queries=queries,
                sides=sides,
                block_rot=block_rot,
                black_padding=black_padding,
                as_obj_only=as_obj_only,
                is_testing=is_testing)
        self.num_gt = 16384
        
    def get_sample_mine(self, idx, debug=False):
        n_parts = self.n_max_parts
        assert n_parts == 2
        depth   = self.pose_dataset.get_depth(idx)
        camintr = self.pose_dataset.get_camintr(idx)
        cloud = self.pose_dataset.get_pcloud(depth, camintr) # * 1000 - center3d
        obj_segm = self.pose_dataset.get_segm(idx, ext_cmd='obj', debug=False) # only TODO
        obj_segm[obj_segm>1.0] = 1.0
        # obj_pcloud = cloud[np.where(obj_segm>0)[0], np.where(obj_segm>0)[1], :] # object cloud
        obj_hand_segm = (np.asarray(self.pose_dataset.get_segm(idx, debug=False)) / 255).astype(np.int)
        segm = obj_hand_segm[:, :, 0] | obj_hand_segm[:, :, 1]

        model_path = self.pose_dataset.obj_paths[idx].replace(
            "model_normalized.pkl", "surface_points.pkl"
        )
        with open(model_path, "rb") as obj_f:
            canon_pts = pickle.load(obj_f)

        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]

        pts_arr = cloud[np.where(segm>0)[0], np.where(segm>0)[1], :]
        cls_arr = obj_segm[np.where(segm>0)[0], np.where(segm>0)[1]] # hand is 0, obj =1
        nocs    = self.pose_dataset.get_nocs(idx, pts_arr, boundary_pts, sym_aligned_nocs=self.cfg.sym_aligned_nocs)
        output_arr     = [pts_arr, cls_arr, nocs]

        # sample only object NOCS points given num_points
        obj_cls    = 1
        n_total_points   = np.where(cls_arr==obj_cls)[0].shape[0]

        if n_total_points < self.num_points:
            tile_n = int(self.num_points/n_total_points) + 1
            n_total_points = tile_n * n_total_points
            for j in range(len(output_arr)):
                arr_tiled = np.concatenate([np.copy(output_arr[j])] * tile_n, axis=0)
                output_arr[j] = arr_tiled
        pts_arr, cls_arr, p_arr = output_arr
        perm       = np.random.permutation(n_total_points)
        obj_inds   = np.where(cls_arr==obj_cls)[0]
        p_arr      = p_arr[obj_inds][perm][:self.num_points]

        # instance 16384 full pts
        gt_points, _ = self.pose_dataset.get_surface_pts(idx, canon_pts=canon_pts)
        gt_points = np.random.permutation(gt_points)
        gt_points = gt_points[:self.num_gt, :]

        # also get category, instance_id
        model_path    = self.pose_dataset.obj_paths[idx]
        category_name = model_path.split('/')[-4]
        instance_name = model_path.split('/')[-3]

        # mask_array    = np.zeros([self.num_points, n_parts], dtype=np.float32)
        # mask_array[np.arange(self.num_points), cls_arr.astype(np.int8)] = 1.00 #
        # mask_array[np.arange(self.num_points), 0] = 0.0 # will not consider hands
        # mask_array[np.arange(self.num_points), n_parts-1] = 0.0 # ignore hand points for nocs prediction
        if not self.is_testing:
            # pts_arr = torch.from_numpy(pts_arr.astype(np.float32).transpose(1, 0))
            # cls_arr = torch.from_numpy(cls_arr.astype(np.float32))
            # mask_array = torch.from_numpy(mask_array.astype(np.float32).transpose(1, 0))
            p_arr = torch.from_numpy(p_arr.astype(np.float32))
            gt_points = torch.from_numpy(gt_points.astype(np.float32))

        return category_name, instance_name, p_arr, gt_points

    def __getitem__(self, idx):
        try:
            idx = self.all_ids[idx]
            sample = self.get_sample_mine(idx)

        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self))
            idx = self.all_ids[random_idx]
            sample = self.get_sample_mine(idx)
        return sample

    def display_nocs(self, ax, sample, proj="z", joint_idxs=False, axis_off=False):
        # Scatter  projection of 3d vertices
        pts = []
        if TransQueries.nocs in sample:
            verts3d = sample[TransQueries.nocs]
            pts.append(verts3d)
            ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], s=1)

        # Scatter  projection of 3d vertices
        if BaseQueries.nocs in sample:
            verts3d = sample[BaseQueries.nocs]
            pts.append(verts3d)
            ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], s=1)

        if 'canon_pts' in sample:
            obj_verts3d = sample['canon_pts']
            pts.append(obj_verts3d)
            ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

        cam_equal_aspect_3d(ax, np.concatenate(pts, axis=0))
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        if axis_off:
            plt.axis('off')

    def visualize_3d_transformed(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.images,
            BaseQueries.occupancy,
            BaseQueries.sides,
            BaseQueries.objpoints3d,
            BaseQueries.verts3d,
            BaseQueries.joints2d,
            BaseQueries.joints3d,
            BaseQueries.depth,
            BaseQueries.pcloud,
            BaseQueries.nocs,
            TransQueries.joints3d,
            TransQueries.images,
            TransQueries.objpoints3d,
            TransQueries.verts3d,
            TransQueries.nocs,
        ]
        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)
        sample = self.get_sample(idx, query=sample_queries, debug=True)
        # print('sample has ', sample.keys())
        # self.save_for_viz(sample, index=idx)
        # vis_utils.visualize_pointcloud(sample['inputs'], title_name='inputs', backend='pyrender')
        # vis_utils.visualize_pointcloud(sample['points'], title_name='points', labels=sample['points.occ'], backend='pyrender')
        # img = sample[TransQueries.images].numpy().transpose(1, 2, 0)

        canon_pts = sample['canon_pts']
        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        canon_pts_normalized = (canon_pts - center_pt.reshape(1, 3)) / length_bb + 0.5
        # vis_utils.visualize_pointcloud([sample[BaseQueries.pcloud], sample['full_pcloud']], title_name='input depth', backend='pyrender')
        # vis_utils.visualize_pointcloud([sample[TransQueries.verts3d], sample[TransQueries.objpoints3d]], title_name='verts & object', backend='pyrender')
        vis_utils.visualize_pointcloud([sample[BaseQueries.nocs], canon_pts_normalized], title_name='points', labels=sample['points.occ'], backend='pyrender')
        # # Display XY projection
        # ax = fig.add_subplot(142)
        # self.display_proj(ax, sample, proj="z", joint_idxs=joint_idxs)
        #
        # # Display YZ projection
        # ax = fig.add_subplot(121, projection='3d')
        # self.display_nocs(ax, sample, proj="y", joint_idxs=joint_idxs)
        #
        # # Display XZ projection
        # ax = fig.add_subplot(122, projection='3d')
        # self.display_3d(ax, sample, proj="y", joint_idxs=joint_idxs)
        #
        # show 3d points
