import numpy as np
import random
import os
import h5py
import pickle
import argparse
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import hydra
import scipy
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf

import __init__
from common.data_utils import get_model_pts, write_pointcloud, get_urdf, split_dataset, get_urdf_mobility
from common.d3_utils import point_3d_offset_joint
from common.vis_utils import plot3d_pts, plot_arrows, plot_imgs
from common.transformations import euler_matrix
from global_info import global_info
epsilon = 10e-8

infos           = global_info()
base_path       = infos.base_path
group_dir       = infos.group_path
second_path     = infos.second_path

def breakpoint():
    import pdb;pdb.set_trace()

class SyntheticDataset(Dataset):
    def __init__(self, root_dir, ctgy_obj, mode, n_max_parts, batch_size, name_dset='shape2motion', num_expr=0.01, domain=None, nocs_type='A', parametri_type='orthogonal', first_n=-1,  \
                   add_noise=False, fixed_order=False, is_debug=False, is_testing=False, is_gen=False, baseline_joints=False):
        self.root_dir     = root_dir
        self.second_dir   = root_dir
        self.name_dset    = name_dset
        self.ctgy_obj     = ctgy_obj
        infos             = global_info()
        self.ctgy_spec    = infos.datasets[ctgy_obj]
        self.parts_map    = infos.datasets[ctgy_obj].parts_map
        self.baseline_joints = baseline_joints

        self.num_points   = int(2048 * 1) # fixed for category with < 5 parts
        self.J_num        = 21
        self.batch_size   = batch_size
        self.n_max_parts  = n_max_parts
        self.fixed_order  = fixed_order
        self.first_n      = first_n
        self.add_noise    = add_noise
        self.is_testing   = is_testing
        self.is_gen       = is_gen
        self.is_debug     = is_debug
        self.nocs_type    = nocs_type
        self.line_space   = parametri_type
        self.hdf5_file_list = []
        self.rot_mats     = {}

        # >>>>>>>>>>>>>>>>>>>> collect data names & paths
        if mode == 'train':
            idx_txt = self.second_dir + '/splits/{}/{}/train.txt'.format(ctgy_obj, num_expr)
        elif mode == 'demo':
            idx_txt = self.second_dir + '/splits/{}/{}/demo.txt'.format(ctgy_obj, num_expr)
        else:
            idx_txt = self.second_dir + '/splits/{}/{}/test.txt'.format(ctgy_obj, num_expr)

        # >>>>>>>>>>>>>>>>>>>> read into name container
        # Case 1: all data preprocessed into h5/npy
        # Case 2: all data needs to be collected one by one;

        # mode & domain to control train, val, test, demo
        with open(idx_txt, "r", errors='replace') as fp:
            line = fp.readline()
            cnt  = 1
            while line:
                hdf5_file = line.strip()
                item = hdf5_file.split('.')[0].split('/')[-2].split('_')[0] # TODO
                if mode=='test':
                    if domain=='seen' and (item not in infos.datasets[ctgy_obj].test_list):
                        self.hdf5_file_list.append(hdf5_file)
                    if domain=='unseen' and (item in infos.datasets[ctgy_obj].test_list):
                        self.hdf5_file_list.append(hdf5_file)
                    if domain is None:
                        self.hdf5_file_list.append(hdf5_file)
                else:
                    self.hdf5_file_list.append(hdf5_file)

                line = fp.readline()


        print('data numbers: ', len(self.hdf5_file_list), self.hdf5_file_list[0])

        if not fixed_order:
            random.shuffle(self.hdf5_file_list)

        # in case we want to overfit one example
        if first_n != -1:
            self.hdf5_file_list = self.hdf5_file_list[:first_n]

        # name for each seperate data
        self.basename_list = [p.split('.')[0].split('/')[-2]+'_'+p.split('.')[0].split('/')[-1] for p in self.hdf5_file_list]
        print('basename_list: ', self.basename_list[0])

        self.n_data = len(self.hdf5_file_list)
        self.first_iteration_finished = False

        # whole URDF points, load all obj files
        self.all_factors, self.all_corners = self.fetch_factors_nocs(self.ctgy_obj, is_debug=self.is_debug, is_gen=self.is_gen)
        self.all_joints = self.fetch_joints_params(self.ctgy_obj, is_debug=self.is_debug)

    def __getitem__(self, i):
        assert not self.first_iteration_finished
        path = self.hdf5_file_list[i]
        if self.is_testing or self.is_debug:
            print('Fetch {}th datapoint from {}'.format(i, path))

        item = path.split('.')[0].split('/')[-2].split('_')[0]
        norm_factor_instance = self.all_factors[item]
        corner_pts_instance  = self.all_corners[item]
        joints = self.all_joints[item]

        with h5py.File(path, 'r') as handle:
            base_name = self.basename_list[i]
            data = self.create_unit_data_from_hdf5(handle, self.n_max_parts, self.num_points, parts_map=self.parts_map, instance=item, base_name=base_name, \
                                norm_factors=norm_factor_instance, norm_corners=corner_pts_instance, joints=joints, nocs_type=self.nocs_type, \
                                add_noise=self.add_noise, fixed_order=self.fixed_order, shuffle=not self.fixed_order, line_space=self.line_space,\
                                is_testing=self.is_testing)
            data['index'] = i
            # assert data is not None # assume data are all clean

        return data

    def __len__(self):
        """
        Return the length of data here
        """
        return self.n_data

    #>>>>>>>>>>>>>>>>>>>>>> get item from .h5
    def create_unit_data_from_hdf5(self, f, n_max_parts, num_points, parts_map= [[0, 3, 4], [1, 2]], instance=None, base_name=None,\
                                norm_corners=[None], norm_factors=[None], joints=None, nocs_type='A', line_space='orth', thres_r=0.2,\
                                add_noise=False, fixed_order=False, check_only=False, shuffle=True,\
                                is_testing=False, is_debug=False):
        n_parts   = len(parts_map)
        nocs_p, nocs_g, nocs_n, parts_cls, parts_pts, offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params, hand_joints, hand_joints_heatmap, hand_joints_unitvec, hand_joints_cls, n_total_points = self.create_data_shape2motion(f, n_max_parts, num_points, parts_map = parts_map, instance=instance,\
                                    base_name=base_name, norm_corners=norm_corners, norm_factors=norm_factors, joints=joints, nocs_type=nocs_type, line_space=line_space, thres_r=thres_r,\
                                    add_noise=add_noise, fixed_order=fixed_order, check_only=check_only, shuffle=shuffle, \
                                    is_testing=is_testing, is_debug=is_debug)

        # >>>>>>>>>>>>>>>>>>>>> hand related
        J_num = self.J_num
        hand_joints_offset_heatmap = np.zeros([n_total_points, 1*J_num], dtype=np.float32)
        hand_joints_offset_unitvec = np.zeros([n_total_points, 3*J_num], dtype=np.float32)
        hand_joint_cls  = np.zeros([n_total_points], dtype=np.float32)

        hand_joints_offset_heatmap[-parts_pts[-1].shape[0]:] = hand_joints_heatmap
        hand_joints_offset_unitvec[-parts_pts[-1].shape[0]:] = hand_joints_unitvec
        hand_joint_cls[-parts_pts[-1].shape[0]:] = 1 # hand points

        # >>>>>>>>>>>>>>>>>>>> points & seg labels
        pts_arr = np.concatenate(parts_pts, axis=0)
        cls_arr = np.concatenate(parts_cls, axis=0)

        #>>>>>>>>>>>>>>>>>>>>> pts to joints offsets(object)
        offset_heatmap =  np.concatenate(offset_heatmap, axis=0)
        if is_debug:
            print('offset_heatmap max is {}'.format(np.amax(offset_heatmap)))
        offset_unitvec =  np.concatenate(offset_unitvec, axis=0)
        joint_orient   =  np.concatenate(joint_orient, axis=0)
        joint_cls      =  np.concatenate(joint_cls, axis=0)

        #>>>>>>>>>>>>>>>>>>>>> NOCS value
        if nocs_p[0] is not None:
            p_arr = np.concatenate(nocs_p, axis=0)
        if nocs_n[0] is not None:
            n_arr = np.concatenate(nocs_n, axis=0)
        if nocs_g[0] is not None:
            g_arr = np.concatenate(nocs_g, axis=0)

        if np.amax(cls_arr) >= n_parts:
            print('max label {} > n_parts {}'.format(np.amax(cls_arr), n_parts))
            return None

        if n_total_points < num_points:
            tile_n = int(num_points/n_total_points) + 1
            n_total_points = tile_n * n_total_points
            cls_tiled = np.concatenate([cls_arr] * tile_n, axis=0)
            cls_arr = cls_tiled

            pts_tiled = np.concatenate([pts_arr] * tile_n, axis=0)
            pts_arr = pts_tiled

            offset_heatmap_tiled = np.concatenate([offset_heatmap] * tile_n, axis=0)
            offset_heatmap   = offset_heatmap_tiled
            offset_unitvec_tiled = np.concatenate([offset_unitvec] * tile_n, axis=0)
            offset_unitvec   = offset_unitvec_tiled

            joint_orient_tiled = np.concatenate([joint_orient] * tile_n, axis=0)
            joint_orient     = joint_orient_tiled
            joint_cls_tiled  = np.concatenate([joint_cls] * tile_n, axis=0)
            joint_cls        = joint_cls_tiled

            hand_joints_offset_heatmap_tiled = np.concatenate([hand_joints_offset_heatmap] * tile_n, axis=0)
            hand_joints_offset_heatmap   = hand_joints_offset_heatmap_tiled
            hand_joints_offset_unitvec_tiled = np.concatenate([hand_joints_offset_unitvec] * tile_n, axis=0)
            hand_joints_offset_unitvec   = hand_joints_offset_unitvec_tiled
            hand_joint_cls_tiled  = np.concatenate([joint_cls] * tile_n, axis=0)
            hand_joint_cls        = hand_joint_cls_tiled
            if nocs_p[0] is not None:
                p_tiled = np.concatenate([p_arr] * tile_n, axis=0)
                p_arr   = p_tiled

            if nocs_n[0] is not None:
                n_tiled = np.concatenate([n_arr] * tile_n, axis=0)
                n_arr   = n_tiled

            if nocs_g[0] is not None:
                g_tiled = np.concatenate([g_arr] * tile_n, axis=0)
                g_arr   = g_tiled

        if check_only:
            return True

        mask_array    = np.zeros([num_points, n_parts], dtype=np.float32)

        # use furthest point sampling
        perm       = np.random.permutation(n_total_points)
        sample_ind = np.arange(0, self.num_points).tolist()
        # sample_ind    = farthest_point_sample(self.num_points, pts_arr[perm][np.newaxis, :, :])
        # sample_ind = sample_ind.eval()[0, :]

        pts_arr       = pts_arr[perm][sample_ind] * norm_factors[0] # by default is 1
        cls_arr       = cls_arr[perm][sample_ind]
        mask_array[np.arange(num_points), cls_arr.astype(np.int8)] = 1.00
        mask_array[np.arange(num_points), n_parts-1] = 0.0 # ignore hand points for nocs prediction

        offset_heatmap_arr = offset_heatmap[perm][sample_ind]
        offset_unitvec_arr = offset_unitvec[perm][sample_ind]
        joint_orient_arr   = joint_orient[perm][sample_ind]
        joint_cls_arr  = joint_cls[perm][sample_ind]
        joint_cls_mask = np.zeros((joint_cls_arr.shape[0]), dtype=np.float32)
        id_valid       = np.where(joint_cls_arr>0)[0]
        joint_cls_mask[id_valid] = 1.00

        hand_joints_offset_heatmap = hand_joints_offset_heatmap[perm][sample_ind]
        hand_joints_offset_unitvec = hand_joints_offset_unitvec[perm][sample_ind]
        hand_joint_cls = hand_joint_cls[perm][sample_ind]

        if nocs_p[0] is not None:
            p_arr = p_arr[perm][sample_ind]
        if nocs_n[0] is not None:
            n_arr = n_arr[perm][sample_ind]
        if nocs_g[0] is not None:
            g_arr = g_arr[perm][sample_ind]

        # rotate according to urdf_ins joint_rpy
        if self.name_dset == 'sapien':
            target_order = self.ctgy_spec.spec_map[instance]
            joint_rpy = joints['joint']['rpy'][target_order[0]]
            rot_mat = euler_matrix(joint_rpy[0], joint_rpy[1], joint_rpy[2])[:3, :3]
            if nocs_p[0] is not None:
                p_arr   = np.dot(p_arr-0.5, rot_mat.T) + 0.5
            g_arr   = np.dot(g_arr-0.5, rot_mat.T) + 0.5
            offset_unitvec_arr= np.dot(offset_unitvec_arr, rot_mat.T)
            joint_orient_arr  = np.dot(joint_orient_arr, rot_mat.T)

        hand_joints = hand_joints.reshape(-1)

        if not self.is_testing:
            pts_arr = torch.from_numpy(pts_arr.astype(np.float32).transpose(1, 0))
            cls_arr = torch.from_numpy(cls_arr.astype(np.float32))
            mask_array = torch.from_numpy(mask_array.astype(np.float32).transpose(1, 0))
            p_arr = torch.from_numpy(p_arr.astype(np.float32).transpose(1, 0))
            g_arr = torch.from_numpy(g_arr.astype(np.float32).transpose(1, 0))
            hand_joints = torch.from_numpy(hand_joints.astype(np.float32))

            hand_joints_offset_heatmap = torch.from_numpy(hand_joints_offset_heatmap.astype(np.float32).transpose(1, 0))
            hand_joints_offset_unitvec = torch.from_numpy(hand_joints_offset_unitvec.astype(np.float32).transpose(1, 0))
            hand_joint_cls = torch.from_numpy(hand_joint_cls.astype(np.float32))

            offset_heatmap_arr = torch.from_numpy(offset_heatmap_arr.astype(np.float32))
            offset_unitvec_arr = torch.from_numpy(offset_unitvec_arr.astype(np.float32).transpose(1, 0))
            joint_orient_arr = torch.from_numpy(joint_orient_arr.astype(np.float32).transpose(1, 0))
            joint_cls_arr = torch.from_numpy(joint_cls_arr.astype(np.float32))
            joint_cls_mask = torch.from_numpy(joint_cls_mask.astype(np.float32))
            joint_params = torch.from_numpy(joint_params)

        result = {
            'P': pts_arr,
            'partcls_per_point': cls_arr, #
            'part_mask': mask_array,
            'nocs_per_point' : p_arr,
            'gocs_per_point' : g_arr,
            'regression_params': hand_joints, # 21, 3
            'handheatmap_per_point': hand_joints_offset_heatmap,
            'handunitvec_per_point': hand_joints_offset_unitvec,
            'hand_mask' : hand_joint_cls,
            'heatmap_per_point'   : offset_heatmap_arr,
            'unitvec_per_point'   : offset_unitvec_arr,
            'orient_per_point'    : joint_orient_arr,
            'jointcls_per_point' : joint_cls_arr,
            'joint_mask'  : joint_cls_mask,
            'joint_params' : joint_params,
        }

        return result

    def create_data_shape2motion(self, f, n_max_parts, num_points, parts_map = [[0, 3, 4], [1, 2]], instance=None, \
                                base_name=None, norm_corners=[None], norm_factors=[None], joints=None, nocs_type='A', line_space='orth', thres_r=0.2,\
                                add_noise=False, fixed_order=False, check_only=False, shuffle=True, \
                                is_testing=False, is_debug=False):
        '''
            f will be a h5py group-like object
        '''
        # >>>>>>>>>>>>>>>>> create container <<<<<<<<<<<<<<<<< #
        n_parts   = len(parts_map)  # parts map to combine points
        parts_pts = [None] * n_parts
        parts_gts = [None] * n_parts
        parts_cls = [None] * n_parts
        parts_nocs= [None] * n_parts
        nocs_p    = [None] * n_parts
        nocs_g    = [None] * n_parts
        nocs_n    = [None] * n_parts
        n_total_points = 0
        parts_parent_joint= [None] * n_parts
        parts_child_joint = [None] * n_parts
        if n_parts ==2:
            parts_offset_joint= [[], []]
            parts_joints      = [[], []]
            joint_index       = [[], []]
        elif n_parts == 3:
            parts_offset_joint= [[], [], []] # per part pts offsets to correponding parts
            parts_joints      = [[], [], []] # joint params list of the joints
            joint_index       = [[], [], []] # joint index recording the corresponding parts
        elif n_parts == 4:
            parts_offset_joint= [[], [], [], []] # per part pts offsets to correponding parts
            parts_joints      = [[], [], [], []] # joint params list of the joints
            joint_index       = [[], [], [], []] # joint index recording the corresponding parts

        # urdf joints properties
        joint_xyz = joints['link']['xyz']
        joint_rpy = joints['joint']['axis']
        joint_part= joints['joint']['parent']
        joint_type= joints['joint']['type']

        joint_params = np.zeros((n_parts, 7))
        if line_space == 'plucker':
            joint_params = np.zeros((n_parts, 6))

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> read h5 file <<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
        hand_joints   = f['joints'][()]
        hand_contacts = f['contacts'][()]
        for idx, group in enumerate(parts_map):
            P = f['gt_points'][str(group[0])][()][:, :3]
            for i in range(1, len(group)):
                P = np.concatenate((P, f['gt_points'][str(group[i])][()][:, :3]), axis=0)
            parts_pts[idx] = P
            n_total_points += P.shape[0]

            parts_cls[idx] = idx * np.ones((P.shape[0]), dtype=np.float32)
            Pc = f['gt_coords'][str(group[0])][()][:, :3]
            for i in range(1, len(group)):
                Pc = np.concatenate((Pc, f['gt_coords'][str(group[i])][()][:, :3]), axis=0)
            parts_gts[idx] = Pc
            parts_parent_joint[idx] = group[0] # first element as part that serve as child
            parts_child_joint[idx] = [ind for ind, x in enumerate(joint_part) if x == group[-1]] # in a group, we may use the last element to find joint that part serves as parent

        # #>>>>>>>>>>>>>>>>>>>>>>>> obb in camera space, normalize the hand
        mean_hand  = np.mean(parts_pts[-1], axis=0).reshape(1, 3)
        # print('rot_mats are ', self.rot_mats)
        # if base_name in self.rot_mats.keys():
        #     rot_mat = self.rot_mats[base_name]
        # else:
        #     print('computing raw svd for ', base_name)
        #     UU, SS, VV = scipy.linalg.svd(parts_pts[-1])
        #     rot_mat = VV.T
        #     self.rot_mats[base_name] = rot_mat
        #     print('adding to rot_mats')
        #
        # for j in range(n_parts):
        #     parts_pts[j] = np.sum(np.expand_dims(parts_pts[j], 2) * rot_mat[:3, :3], axis=1)
        #     parts_pts[j] = parts_pts[j] - mean_hand
        # hand_joints   = np.sum(np.expand_dims(hand_joints, 2) * rot_mat[:3, :3], axis=1) - mean_hand
        # hand_contacts = np.sum(np.expand_dims(hand_contacts, 2) * rot_mat[:3, :3], axis=1) - mean_hand
        for j in range(n_parts):
            # parts_pts[j] = np.dot(parts_pts[j], rot_mat)
            parts_pts[j] = parts_pts[j] - mean_hand
        # hand_joints   = np.dot(hand_joints, rot_mat) - mean_hand
        # hand_contacts = np.dot(hand_contacts, rot_mat)- mean_hand
        hand_joints   = hand_joints - mean_hand
        hand_contacts = hand_contacts - mean_hand

        # >>>>>>>>>>>>>>>>>>> compute NOCS, joint_params, associated joints for each part, ignore joints for hand
        for j in range(n_parts):
            nocs_p[j] = parts_gts[j][:, :3] # but we only supervise on the first 3 parts
            nocs_g[j] = parts_gts[j][:, :3] # TODO

            if j == n_parts - 1:
                continue

            norm_factor = norm_factors[j+1]
            norm_corner = norm_corners[j+1]
            if j>0:
                joint_P0  = - np.array(joint_xyz[j])
                joint_P0  = (joint_P0 - norm_corner[0]) * norm_factor  + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
                joint_l   = np.array(joint_rpy[j])
                orth_vect = point_3d_offset_joint([joint_P0, joint_l], np.array([0, 0, 0]).reshape(1, 3))
                joint_params[j, 0:3] = joint_l
                joint_params[j, 6]   = np.linalg.norm(orth_vect)
                joint_params[j, 3:6] = orth_vect/joint_params[j, 6]

            if parts_parent_joint[j] !=0:
                joint_P0  = - np.array(joint_xyz[parts_parent_joint[j]])
                joint_P0  = (joint_P0 - norm_corner[0]) * norm_factor  + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
                joint_l   = np.array(joint_rpy[j])
                offset_arr= point_3d_offset_joint([joint_P0, joint_l], nocs_g[j])
                parts_offset_joint[j].append(offset_arr)
                parts_joints[j].append([joint_P0, joint_l])
                joint_index[j].append(parts_parent_joint[j])
                # if is_debug:
                #     plot_arrows(nocs_g[j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, j))
            if parts_child_joint[j] is not None:
                for m in parts_child_joint[j]:
                    joint_P0  = - np.array(joint_xyz[m])
                    joint_P0  = (joint_P0 - norm_corner[0]) * norm_factor  + np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (  norm_corner[1] - norm_corner[0]) * norm_factor
                    joint_l   = np.array(joint_rpy[m])
                    offset_arr= point_3d_offset_joint([joint_P0, joint_l], nocs_g[j])
                    parts_offset_joint[j].append(offset_arr)
                    parts_joints[j].append([joint_P0, joint_l])
                    joint_index[j].append(m)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> get hand_joints_offsets, and related infos
        J_num = self.J_num
        hand_joints_heatmap = np.zeros((parts_pts[-1].shape[0], J_num, 1), dtype=np.float32)
        hand_joints_unitvec = np.zeros((parts_pts[-1].shape[0], J_num, 3), dtype=np.float32)
        hand_offsets = parts_pts[-1][:, np.newaxis, :] - hand_joints[np.newaxis, :, :] # 1, 21, 3 --> N, 1, 3
        hand_heatmap = np.linalg.norm(hand_offsets, axis=2)
        hand_unitvec = -hand_offsets/(hand_heatmap[:, :, np.newaxis] + epsilon)
        hand_joints_cls = np.zeros((parts_pts[-1].shape[0], J_num), dtype=np.float32)

        idc          = np.where(hand_heatmap<thres_r)
        hand_joints_heatmap[idc[0], idc[1], 0] = 1 - hand_heatmap[idc[0], idc[1]]/thres_r # otthers are 0
        hand_joints_unitvec[idc[0], idc[1], :] = hand_unitvec[idc[0], idc[1], :]
        hand_joints_heatmap = hand_joints_heatmap.reshape(-1, 1*J_num)
        hand_joints_unitvec = hand_joints_unitvec.reshape(-1, 3*J_num)
        hand_joints_cls[idc[0], idc[1]] = 1

        offset_heatmap = [None] * (n_parts)
        offset_unitvec = [None] * (n_parts)
        joint_orient   = [None] * (n_parts)
        joint_cls      = [None] * (n_parts)

        for j, offsets in enumerate(parts_offset_joint):
            offset_heatmap[j] = np.zeros((parts_gts[j].shape[0]))
            offset_unitvec[j] = np.zeros((parts_gts[j].shape[0], 3))
            joint_orient[j]   = np.zeros((parts_gts[j].shape[0], 3))
            joint_cls[j]      = np.zeros((parts_gts[j].shape[0]))

            for k, offset in enumerate(offsets):
                heatmap = np.linalg.norm(offset, axis=1)
                unitvec = offset/(heatmap.reshape(-1, 1) + epsilon)
                idc     = np.where(heatmap<thres_r)[0]
                offset_heatmap[j][idc]    = 1 - heatmap[idc]/thres_r

                offset_unitvec[j][idc, :] = unitvec[idc, :]
                joint_orient[j][idc, :]   = parts_joints[j][k][1]
                joint_cls[j][idc]         = joint_index[j][k]

            if is_debug:
                plot_arrows_list(nocs_g, parts_offset_joint, parts_joints, title_name='joint offset')
                plot_arrows_list_threshold(nocs_g, parts_offset_joint, parts_joints, title_name='joint offset')

        return  nocs_p, nocs_g, nocs_n, parts_cls, parts_pts, \
                offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params, \
                hand_joints, hand_joints_heatmap, hand_joints_unitvec, hand_joints_cls, n_total_points

    # >>>>>>>>>>>>>>>>> require 'urdf' or 'objects', usually 'pickle' would be enough
    def fetch_factors_nocs(self, obj_category, is_debug=False, is_gen=False):
        if is_gen:
            all_items   = os.listdir(self.root_dir + '/render/' + obj_category) # check according to render folder
            all_factors = {}
            all_corners = {}
            pts_m       = {}
            root_dset   = self.root_dir
            offsets     = None
            for item in all_items:
                if self.name_dset == 'sapien':
                    path_urdf = self.root_dir + '/objects/' + '/' + obj_category + '/' + item
                    urdf_ins   = get_urdf_mobility(path_urdf)
                elif self.name_dset == 'shape2motion':
                    path_urdf = self.root_dir + '/urdf/' + '/' + obj_category
                    urdf_ins   = get_urdf("{}/{}".format(path_urdf, item))
                else:
                    path_urdf = self.root_dir + '/urdf/' + '/' + obj_category
                    urdf_ins   = get_urdf("{}/{}".format(path_urdf, item))
                pts, norm_factors, corner_pts = get_model_pts(self.root_dir, self.ctgy_obj, item, obj_file_list=urdf_ins['obj_name'],  offsets=offsets , is_debug=is_debug)
                all_factors[item]        = norm_factors
                all_corners[item]        = corner_pts

                pt_ii           = []
                bbox3d_per_part = []
                for p, pt in enumerate(pts):
                    pt_s = np.concatenate(pt, axis=0)
                    pt_ii.append(pt_s)
                    print('We have {} pts'.format(pt_ii[p].shape[0]))
                if pt_ii is not []:
                    pts_m[item] = pt_ii
                else:
                    print('!!!!! {} model loading is wrong'.format(item))
            # save into pickle file, need to make pickle folder
            directory = root_dset + "/pickle/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(root_dset + "/pickle/{}.pkl".format(obj_category), "wb") as f:
                pickle.dump(all_factors, f)
            with open(root_dset + "/pickle/{}_corners.pkl".format(obj_category), "wb") as fc:
                pickle.dump(all_corners, fc)
            with open(root_dset + "/pickle/{}_pts.pkl".format(obj_category), 'wb') as fp:
                pickle.dump(pts_m, fp)
        else:
            root_dset   = self.root_dir
            file = open(root_dset + "/pickle/{}.pkl".format(obj_category),"rb")
            data = pickle.load(file)
            all_factors = data
            file.close()
            fc = open(root_dset + "/pickle/{}_corners.pkl".format(obj_category), "rb")
            all_corners = pickle.load(fc)
            fc.close()

        return all_factors, all_corners

    #>>>>>>>>>>>>>>>>>>>>>>> require 'urdf' or 'objects' folder
    def fetch_joints_params(self, obj_category, is_debug=False):
        all_items   = os.listdir(self.root_dir + '/urdf/' + obj_category) #TODO: which one to choose? urdf or render?
        all_joints  = {}
        root_dset   = self.root_dir
        for item in all_items:
            if self.name_dset == 'shape2motion':
                path_urdf = self.root_dir + '/urdf/' + '/' + obj_category
                urdf_ins   = get_urdf("{}/{}".format(path_urdf, item))
            elif self.name_dset == 'sapien':
                path_urdf = self.root_dir + '/objects/' + '/' + obj_category + '/' + item
                urdf_ins   = get_urdf_mobility(path_urdf)
            else:
                path_urdf = self.root_dir + '/urdf/' + '/' + obj_category
                urdf_ins   = get_urdf("{}/{}".format(path_urdf, item))
            if obj_category == 'bike':
                urdf_ins['link']['xyz'][1], urdf_ins['link']['xyz'][2] = urdf_ins['link']['xyz'][2], urdf_ins['link']['xyz'][1]
                urdf_ins['joint']['axis'][1], urdf_ins['joint']['axis'][2] = urdf_ins['joint']['axis'][2], urdf_ins['joint']['axis'][1]
            all_joints[item] = urdf_ins

            if is_debug:
                print(urdf_ins['link']['xyz'], urdf_ins['joint']['axis'])

        return all_joints

@hydra.main(config_path="../config/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(cfg.port)
    infos       = global_info()
    data_infos  = infos.datasets[cfg.item]
    root_dset   = cfg.root_data
    # category-wise training setup
    cfg.n_max_parts = data_infos.num_parts
    cfg.is_test = False

    random.seed(30)

    data_set = SyntheticDataset(
        root_dir=cfg.root_data,
        ctgy_obj=cfg.item,
        name_dset=cfg.name_dset,
        batch_size=cfg.batch_size,
        n_max_parts=cfg.n_max_parts,
        add_noise=cfg.train_data_add_noise,
        nocs_type=cfg.nocs_type,
        parametri_type=cfg.parametri_type,
        first_n=cfg.train_first_n,
        is_debug=cfg.is_debug,
        is_testing=True,
        mode='train',
        fixed_order=False,)

    selected_index = np.random.randint(len(data_set.basename_list), size=2)
    # selected_index = [data_set.basename_list.index('0016_0_0_0')] + list(np.arange(0, len(data_set.basename_list)))
    for i in selected_index:
        basename =  data_set.basename_list[i]
        instance = basename.split('_')[0]
        print('reading data point: ', i, data_set.basename_list[i])
        data_pts=  data_set.__getitem__(i)

        for keys, item in data_pts.items():
            print(keys, item.shape)

        input_pts = data_pts['P']
        nocs_gt   = {}
        nocs_gt['pn']   = data_pts['nocs_per_point']
        nocs_gt['gn']   = data_pts['gocs_per_point']

        mask_gt   = data_pts['partcls_per_point']
        num_pts = input_pts.shape[0]
        num_parts = cfg.n_max_parts
        part_idx_list_gt   = []
        for j in range(num_parts):
            part_idx_list_gt.append(np.where(mask_gt==j)[0])
        if not cfg.test:
            heatmap_gt= data_pts['heatmap_per_point']
            unitvec_gt= data_pts['unitvec_per_point']
            orient_gt = data_pts['orient_per_point']
            joint_cls_gt = data_pts['jointcls_per_point']
            joint_params_gt = data_pts['joint_params']
            joint_idx_list_gt   = []
            for j in range(num_parts):
                joint_idx_list_gt.append(np.where(joint_cls_gt==j)[0])

        #>>>>>>>>>>>>>>>>>>>>>------ For segmentation visualization ----
        # plot_imgs([rgb_img], ['rgb img'], title_name='RGB', sub_name=str(i), show_fig=cfg.show_fig, save_fig=cfg.save_fig, save_path=root_dset + '/NOCS/' + cfg.item)
        plot3d_pts([[input_pts]], [['Part {}'.format(0)]], s=50, title_name=['GT seg on input point cloud'], sub_name=str(i), show_fig=cfg.show_fig, axis_off=False, save_fig=cfg.save_fig, save_path=root_dset + '/NOCS/' + cfg.item, limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]])
        plot3d_pts([[input_pts[part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=50, title_name=['GT seg on input point cloud'], sub_name=str(i), show_fig=cfg.show_fig, axis_off=False, save_fig=cfg.save_fig, save_path=root_dset + '/NOCS/' + cfg.item, limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]])
        plot3d_pts([[nocs_gt['gn'][part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=50, title_name=['GT global NOCS'], sub_name=str(i), show_fig=cfg.show_fig, save_fig=cfg.save_fig, save_path=root_dset + '/NOCS/' + cfg.item)
        plot3d_pts([[nocs_gt['pn'][part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=50, title_name=['GT part NOCS'], sub_name=str(i), show_fig=cfg.show_fig, save_fig=cfg.save_fig, save_path=root_dset + '/NOCS/' + cfg.item)
        for j in range(num_parts):
            plot3d_pts([[nocs_gt['pn'][part_idx_list_gt[j], :] ]], [['Part {}'.format(j)]], s=15, dpi=200, title_name=['GT part NOCS'], color_channel=[[ nocs_gt['pn'][part_idx_list_gt[j], :] ]], sub_name='{}_part_{}'.format(i, j) , show_fig=cfg.show_fig, save_fig=cfg.save_fig, save_path=root_dset + '/NOCS/' + cfg.item)
            plot3d_pts([[nocs_gt['gn'][part_idx_list_gt[j], :] ]], [['Part {}'.format(j)]], s=15, dpi=200, title_name=['GT global NOCS'], color_channel=[[ nocs_gt['gn'][part_idx_list_gt[j], :] ]], sub_name='{}_part_{}'.format(i, j), show_fig=cfg.show_fig, save_fig=cfg.save_fig, save_path=root_dset + '/NOCS/' + cfg.item)

        # show joints
        if not cfg.is_test:
            plot3d_pts([[input_pts[joint_idx_list_gt[j], :] for j in range(num_parts)]], [['unassigned Pts'] + ['Pts of joint {}'.format(j) for j in range(1, num_parts)]], s=15, \
                  title_name=['GT association of pts to joints '], sub_name=str(i), show_fig=cfg.show_fig, save_fig=cfg.save_fig, save_path=root_dset + '/NOCS/' + cfg.item, axis_off=True)
            plot3d_pts([[ input_pts ]], [['Part 0-{}'.format(num_parts-1)]], s=15, \
                  dpi=200, title_name=['Input Points distance heatmap'], color_channel=[[250*np.concatenate([heatmap_gt.reshape(-1, 1), np.zeros((heatmap_gt.shape[0], 2))], axis=1)]], show_fig=cfg.show_fig)

            thres_r       = 0.2
            offset        = unitvec_gt * (1- heatmap_gt.reshape(-1, 1)) * thres_r
            joint_pts     = nocs_gt['gn'] + offset
            joints_list   = []
            idx           = np.where(joint_cls_gt > 0)[0]
            plot_arrows(nocs_gt['gn'][idx], [0.5*orient_gt[idx]], whole_pts=nocs_gt['gn'], title_name='{}_joint_pts_axis'.format('GT'), dpi=200, s=25, thres_r=0.1, show_fig=cfg.show_fig, sparse=True, save=cfg.save_fig, index=i, save_path=root_dset + '/NOCS/' + cfg.item)
            plot_arrows(nocs_gt['gn'][idx], [offset[idx]], whole_pts=nocs_gt['gn'], title_name='{}_joint_offsets'.format('GT'), dpi=200, s=25, thres_r=0.1, show_fig=cfg.show_fig, sparse=True, save=cfg.save_fig, index=i, save_path=root_dset + '/NOCS/' + cfg.item)
            # plot_arrows(input_pts[idx], [0.5*orient_gt[idx]], whole_pts=input_pts, title_name='{}_joint_pts_axis'.format('GT'), dpi=200, s=25, thres_r=0.2, show_fig=cfg.show_fig, sparse=True, save=cfg.save_fig, index=i, save_path=root_dset + '/NOCS/' + cfg.item)
            # plot_arrows(input_pts[idx], [offset[idx]], whole_pts=input_pts, title_name='{}_joint_offsets'.format('GT'), dpi=200, s=25, thres_r=0.2, show_fig=cfg.show_fig, sparse=True, save=cfg.save_fig, index=i, save_path=root_dset + '/NOCS/' + cfg.item)

if __name__=='__main__':
    main()
