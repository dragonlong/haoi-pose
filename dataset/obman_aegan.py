import random
import pickle
import traceback
from tqdm import tqdm
import numpy as np
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

class HandDatasetAEGan(HandDataset):
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
        self.num_gt = 2048
        # find all unique files
        target = [pose_dataset.obj_paths[id] for id in self.all_ids]
        self.unique_objs = list(set(target))
        print(f'---we have {len(self.unique_objs)} unique instances')

    def get_sample_mine(self, idx, debug=False):
        model_path = self.unique_objs[idx].replace(
            "model_normalized.pkl", "surface_points.pkl"
        )
        with open(model_path, "rb") as obj_f:
            canon_pts = pickle.load(obj_f)

        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # we care about the nomralized shape in NOCS
        gt_points = (canon_pts - center_pt.reshape(1, 3)) / length_bb + 0.5
        gt_points = np.random.permutation(gt_points)
        gt_points = gt_points[:self.num_gt, :]

        # also get category, instance_id
        model_path    = self.pose_dataset.obj_paths[idx]
        category_name = model_path.split('/')[-4]
        instance_name = model_path.split('/')[-3]

        #
        gt_points = torch.from_numpy(gt_points.astype(np.float32).transpose(1, 0))
        return {"points": gt_points, "id": instance_name}

    def get_sample_pair(self, idx, debug=False):
        model_path = self.pose_dataset.obj_paths[idx].replace("model_normalized.pkl", "surface_points.pkl")
        category_name = model_path.split('/')[-4]
        instance_name = model_path.split('/')[-3]
        if idx not in self.nocs_dict:
            n_parts = self.n_max_parts
            assert n_parts == 2
            depth   = self.pose_dataset.get_depth(idx)
            camintr = self.pose_dataset.get_camintr(idx)
            cloud = self.pose_dataset.get_pcloud(depth, camintr) # * 1000 - center3d
            obj_segm = self.pose_dataset.get_segm(idx, ext_cmd='obj', debug=False) # only TODO
            obj_segm[obj_segm>1.0] = 1.0
            obj_hand_segm = (np.asarray(self.pose_dataset.get_segm(idx, debug=False)) / 255).astype(np.int)
            segm = obj_hand_segm[:, :, 0] | obj_hand_segm[:, :, 1]

            with open(model_path, "rb") as obj_f:
                canon_pts = pickle.load(obj_f)

            boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]

            pts_arr = cloud[np.where(segm>0)[0], np.where(segm>0)[1], :]
            cls_arr = obj_segm[np.where(segm>0)[0], np.where(segm>0)[1]] # hand is 0, obj =1
            nocs    = self.pose_dataset.get_nocs(idx, pts_arr, boundary_pts, sym_aligned_nocs=self.cfg.sym_aligned_nocs)
            obj_cls    = 1
            obj_inds   = np.where(cls_arr==obj_cls)[0]
            nocs       = nocs[obj_inds]
            self.nocs_dict[idx] = nocs
        else:
            nocs = self.nocs_dict[idx]

        output_arr     = [nocs]
        n_total_points   = nocs.shape[0]

        if n_total_points < self.num_points:
            tile_n = int(self.num_points/n_total_points) + 1
            n_total_points = tile_n * n_total_points
            for j in range(len(output_arr)):
                arr_tiled = np.concatenate([np.copy(output_arr[j])] * tile_n, axis=0)
                output_arr[j] = arr_tiled
        p_arr      = output_arr[0]
        perm       = np.random.permutation(n_total_points)
        p_arr      = p_arr[perm][:self.num_points]

        # 2048 full pts
        idx1 = self.all_ids[random.randint(0, len(self)-1)]
        model_path = self.pose_dataset.obj_paths[idx1].replace("model_normalized.pkl", "surface_points.pkl")
        category_name1 = model_path.split('/')[-4]
        instance_name1 = model_path.split('/')[-3]
        with open(model_path, "rb") as obj_f:
            canon_pts1 = pickle.load(obj_f)
        boundary_pts = [np.min(canon_pts1, axis=0), np.max(canon_pts1, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # we care about the nomralized shape in NOCS
        gt_points = (canon_pts1 - center_pt.reshape(1, 3)) / length_bb + 0.5
        gt_points = np.random.permutation(gt_points)
        gt_points = gt_points[:self.num_gt, :]

        if not self.is_testing:
            p_arr = torch.from_numpy(p_arr.astype(np.float32).transpose(1, 0))
            gt_points = torch.from_numpy(gt_points.astype(np.float32).transpose(1, 0))
        else:
            return p_arr, gt_points, instance_name, instance_name1
        return {"raw": p_arr, "real": gt_points, "raw_id": instance_name, "real_id": instance_name1}

    def __len__(self):
        if self.task == 'pcloud_completion':
            return len(self.unique_objs)
        else:
            return len(self.all_ids)

    def __getitem__(self, idx):
        try:
            if self.task == 'pcloud_completion':
                sample = self.get_sample_mine(idx)
            else:
                idx = self.all_ids[idx]
                sample = self.get_sample_pair(idx)
        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self)-1)
            idx = self.all_ids[random_idx]
            if self.task == 'pcloud_completion':
                sample = self.get_sample_mine(idx)
            else:
                sample = self.get_sample_pair(idx)

        return sample
