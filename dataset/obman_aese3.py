import random
import pickle
import traceback
from tqdm import tqdm
import numpy as np
import os
from os import makedirs, remove
from os.path import exists, join
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms
import dgl

import __init__
from global_info import global_info
from dataset.obman_handataset import HandDataset
from common import data_utils, handutils, vis_utils, bp
from common.aligning import estimateSimilarityTransform, estimateSimilarityUmeyama
from common.d3_utils import align_rotation, rotate_about_axis, transform_pcloud
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
group_path      = infos.group_path
project_path    = infos.project_path
categories_id   = infos.categories_id

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

def index_points(points, idx):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class FixedRadiusNearNeighbors(nn.Module):
    '''
    Ball Query - Find the neighbors with-in a fixed radius
    '''
    def __init__(self, radius, n_neighbor, knn=False):
        super(FixedRadiusNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor
        self.knn = knn

    def forward(self, pos, centroids):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        '''
        device = pos.device
        B, N, _ = pos.shape
        # center_pos = index_points(pos, centroids)
        center_pos = pos
        _, S, _ = center_pos.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        # print('before square_distance ', center_pos.shape, pos.shape)
        sqrdists = square_distance(center_pos, pos)
        if self.knn:
            _, group_idx = torch.topk(sqrdists, self.n_neighbor+1, dim=-1, largest=False, sorted=True)
            group_idx = group_idx[:, :, 1:self.n_neighbor+1]
        else:
            group_idx[sqrdists > self.radius ** 2] = N
            group_idx = group_idx.sort(dim=-1)[0][:, :, :self.n_neighbor]
            group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.n_neighbor])
            mask = group_idx == N
            group_idx[mask] = group_first[mask]
        return group_idx

class HandDatasetAEGraph(HandDataset):
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
        self.num_gt = cfg.num_points
        self.radius = 0.1
        self.num_samples = 10 # default to be 10
        self.augment = cfg.augment
        self.fetch_cache = cfg.fetch_cache
        # find all unique files
        target = [pose_dataset.obj_paths[id] for id in self.all_ids]
        self.unique_objs = list(set(target))
        print(f'---we have {len(self.unique_objs)} unique instances')
        self.frnn  = FixedRadiusNearNeighbors(self.radius, self.num_samples, knn=True)
        if self.cfg.use_preprocess:
            preprocessed_file = f"{group_path}/external/obman/obman/preprocessed/full_{split}_1024.npy"
            print('loading ', preprocessed_file)
            self.data_dict = np.load(preprocessed_file, allow_pickle=True).item()
            self.all_keys  = list(self.data_dict.keys())

    def get_sample_mine(self, idx, verbose=False):
        """used only for points data"""
        if self.augment and idx!=0:
            theta_x = random.randint(0, 180)
            Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
            theta_z = random.randint(0, 180)
            Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
            RR = torch.from_numpy(np.matmul(Rx, Rz).astype(np.float32))
            TT = torch.from_numpy(np.random.rand(1,3).astype(np.float32))
            # print(f'---{theta_x}x + {theta_z}z rotation')
        else:
            # print('---0 rotation')
            RR = torch.from_numpy(rotate_about_axis(0, axis='x').astype(np.float32))
            TT = 0
        if self.cfg.single_instance:
            # print('using same idx')
            idx = 2
        model_path    = self.unique_objs[idx]
        category_name = model_path.split('/')[-4]
        instance_name = model_path.split('/')[-3]

        # preprocess
        if self.fetch_cache and idx in self.g_dict:
            # print('using cache!!!')
            gt_points, src, dst = self.g_dict[idx]
            pos = gt_points.clone().detach().unsqueeze(0)
            feat= torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))
        else:
            if self.cfg.use_preprocess: # 2. use disk saved results
                gt_points = self.data_dict[instance_name][0]
                group_idx = torch.from_numpy(self.data_dict[instance_name][1].astype(np.int)[:, 1:self.num_samples+1])
                gt_points = torch.from_numpy(gt_points.astype(np.float32)[:, :]) # keep N, 3

                pos = gt_points.clone().detach().unsqueeze(0)
                feat= torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))
                src = group_idx.contiguous().view(-1)
                centroids = torch.from_numpy(np.arange(gt_points.shape[0]).reshape(1, -1))
                dst = centroids[0].view(-1, 1).repeat(1, self.num_samples).view(-1)
            else:
                model_path = self.unique_objs[idx].replace(
                    "model_normalized.pkl", "surface_points.pkl"
                )
                with open(model_path, "rb") as obj_f:
                    canon_pts = pickle.load(obj_f) #

                boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
                center_pt = (boundary_pts[0] + boundary_pts[1])/2
                length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

                # we care about the nomralized shape in NOCS
                gt_points = (canon_pts - center_pt.reshape(1, 3)) / length_bb + 0.5
                gt_points = np.random.permutation(gt_points)
                gt_points = gt_points[:self.num_gt, :]
                gt_points = torch.from_numpy(gt_points.astype(np.float32)[:, :])

                pos = gt_points.clone().detach().unsqueeze(0)
                centroids = torch.from_numpy(np.arange(gt_points.shape[0]).reshape(1, -1))
                group_idx = self.frnn(pos, centroids)

                src = group_idx[0].contiguous().view(-1)
                dst = centroids[0].view(-1, 1).repeat(1, self.num_samples).view(-1) # real pair
                feat      = torch.from_numpy(np.ones((pos.shape[0], pos.shape[1], 1)).astype(np.float32))

            if self.fetch_cache and idx not in self.g_dict:
                # print('caching!!!')
                self.g_dict[idx] = [gt_points, src, dst]
        center_offset = pos[0].clone().detach()-0.5
        if self.augment:
            pos[0] = torch.matmul(pos[0]-0.5, RR) + 0.5 # to have random rotation
            pos[0] = pos[0] + TT
            center_offset = torch.matmul(center_offset, RR) # N, 3

        # construct a graph for input
        unified = torch.cat([src, dst])
        uniq, inv_idx = torch.unique(unified, return_inverse=True)
        src_idx = inv_idx[:src.shape[0]]
        dst_idx = inv_idx[src.shape[0]:]
        if verbose:
            print('src_idx.shape', src_idx.shape, '\n', src_idx[0:100], '\n', 'dst_idx.shape', dst_idx.shape, '\n', dst_idx[0:100])
        g = dgl.DGLGraph((src_idx, dst_idx))
        g.ndata['x'] = pos[0][uniq]
        g.ndata['f'] = feat[0][uniq].unsqueeze(-1)
        g.edata['d'] = pos[0][dst_idx] - pos[0][src_idx] #[num_atoms,3] but we only supervise the half
        #
        up_axis = torch.matmul(torch.tensor([[0.0, 1.0, 0.0]]).float(), RR)

        # return g, gt_points.transpose(1, 0), instance_name, RR # 3*3
        return g, gt_points.transpose(1, 0), instance_name, up_axis, center_offset #up_axis,

    def get_sample_pair(self, idx, verbose=False):
        if self.cfg.single_instance:
            idx = 2
        # idx Corresponds original full dataset
        model_path = self.pose_dataset.obj_paths[idx].replace("model_normalized.pkl", "surface_points.pkl")
        category_name = model_path.split('/')[-4]
        instance_name = model_path.split('/')[-3]

        # fetch GT canonical points  NOCS, input pts_arr
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
                canon_pts = pickle.load(obj_f) #

            boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]

            pts_arr = cloud[np.where(segm>0)[0], np.where(segm>0)[1], :]
            cls_arr = obj_segm[np.where(segm>0)[0], np.where(segm>0)[1]] # hand is 0, obj =1
            nocs    = self.pose_dataset.get_nocs(idx, pts_arr, boundary_pts, sym_aligned_nocs=self.cfg.sym_aligned_nocs)
            obj_cls    = 1
            obj_inds   = np.where(cls_arr==obj_cls)[0]
            nocs       = nocs[obj_inds]
            pts_arr    = pts_arr[obj_inds]
            self.nocs_dict[idx]  = nocs
            self.cloud_dict[idx] = pts_arr
        else:
            nocs    = self.nocs_dict[idx]
            pts_arr = self.cloud_dict[idx]

        output_arr     = [nocs, pts_arr]
        n_total_points   = nocs.shape[0]

        if n_total_points < self.num_points:
            tile_n = int(self.num_points/n_total_points) + 1
            n_total_points = tile_n * n_total_points
            for j in range(len(output_arr)):
                arr_tiled = np.concatenate([np.copy(output_arr[j])] * tile_n, axis=0)
                output_arr[j] = arr_tiled

        n_arr      = output_arr[0]
        p_arr      = output_arr[1]
        perm       = np.random.permutation(n_total_points)
        n_arr      = n_arr[perm][:self.num_points] # nocs in canonical space, for partial reconstruction only
        p_arr      = p_arr[perm][:self.num_points] # point cloud in camera space

        # 2048 full pts, random picking one for adversarial training
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

        g_sets = []
        for target_pts in [p_arr, gt_points]:
            # create graph
            pos = torch.from_numpy(target_pts.astype(np.float32)).unsqueeze(0)
            centroids = torch.from_numpy(np.arange(target_pts.shape[0]).reshape(1, -1))
            # print(pos.shape, centroids.shape)
            feat      = np.ones((pos.shape[0], pos.shape[1], 1))
            group_idx = self.frnn(pos, centroids)

            i = 0
            N = pos.shape[1]
            center = torch.zeros((N))#.to(dev)
            center[centroids[0]] = 1 # find the chosen query
            src = group_idx[0].contiguous().view(-1) # real pair
            dst = centroids[0].view(-1, 1).repeat(1, self.num_samples).view(-1) # real pair

            unified = torch.cat([src, dst])
            uniq, inv_idx = torch.unique(unified, return_inverse=True)
            src_idx = inv_idx[:src.shape[0]]
            dst_idx = inv_idx[src.shape[0]:]
            if verbose:
                print('src_idx.shape', src_idx.shape, '\n', src_idx[0:100], '\n', 'dst_idx.shape', dst_idx.shape, '\n', dst_idx[0:100])
            g = dgl.DGLGraph((src_idx, dst_idx))
            g.ndata['x'] = pos[0][uniq]
            g.ndata['f'] = torch.from_numpy(feat[0][uniq].astype(np.float32)[:, :, np.newaxis])
            g.edata['d'] = pos[0][dst_idx] - pos[0][src_idx] #[num_atoms,3]
            g_sets.append(g)

        g_raw, g_real = g_sets
        _, r, t, _= estimateSimilarityUmeyama(n_arr, p_arr) # get GT Pose
        center_offset = np.matmul(n_arr - 0.5, r)
        # create sub-graph
        if not self.is_testing:
            n_arr = torch.from_numpy(n_arr.astype(np.float32).transpose(1, 0))
            gt_points = torch.from_numpy(gt_points.astype(np.float32).transpose(1, 0))
            center_offset = torch.from_numpy(center_offset).float()
        else:
            return n_arr, gt_points, instance_name, instance_name1

        if idx not in self.r_dict:
            RR = torch.from_numpy(r.astype(np.float32))
            self.r_dict[idx] = RR
        else:
            RR = self.r_dict[idx]
        up_axis = torch.matmul(torch.tensor([[0.0, 1.0, 0.0]]).float(), RR)

        return g_raw, g_real, n_arr, gt_points, instance_name, instance_name1, up_axis, center_offset

    def __len__(self):
        if 'adversarial' in self.task or 'partial' in self.task:
            return len(self.all_ids)
        else:
            return len(self.unique_objs)

    def __getitem__(self, idx):
        try:
            if 'adversarial' in self.task or 'partial' in self.task:
                idx = self.all_ids[idx]
                sample = self.get_sample_pair(idx)
            else:
                sample = self.get_sample_mine(idx)
        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self)-1)
            if 'adversarial' in self.task or 'partial' in self.task:
                idx        = self.all_ids[random_idx]
                sample = self.get_sample_pair(idx)
            else:
                sample = self.get_sample_mine(idx)
        return sample

def main():
    pass

if __name__ == '__main__':
    main()
