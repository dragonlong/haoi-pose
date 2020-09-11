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
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf

import __init__
from common.data_utils import calculate_factor_nocs, get_model_pts, write_pointcloud, get_urdf, split_dataset, get_urdf_mobility
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

        self.num_points   = int(1024 * 1) # fixed for category with < 5 parts
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
        if mode == 'train':
            idx_txt = self.second_dir + '/splits/{}/{}/train.txt'.format(ctgy_obj, num_expr)
        elif mode == 'demo':
            idx_txt = self.second_dir + '/splits/{}/{}/demo.txt'.format(ctgy_obj, num_expr)
        else:
            idx_txt = self.second_dir + '/splits/{}/{}/test.txt'.format(ctgy_obj, num_expr)
        with open(idx_txt, "r", errors='replace') as fp:
            line = fp.readline()
            cnt  = 1
            while line:
                # todos: test mode
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
        if is_debug:
            print('hdf5_file_list: ', len(self.hdf5_file_list), self.hdf5_file_list[0])
        if not fixed_order:
            random.shuffle(self.hdf5_file_list)
        if first_n != -1:
            self.hdf5_file_list = self.hdf5_file_list[:first_n]

        self.basename_list = [p.split('.')[0].split('/')[-2]+'_'+p.split('.')[0].split('/')[-1] for p in self.hdf5_file_list]

        self.n_data = len(self.hdf5_file_list)

        # whole URDF points, load all obj files
        self.all_factors, self.all_corners = self.fetch_factors_nocs(self.ctgy_obj, is_debug=self.is_debug, is_gen=self.is_gen)
        self.all_joints = self.fetch_joints_params(self.ctgy_obj, is_debug=self.is_debug)

    def __getitem__(self, i):
        path = self.hdf5_file_list[i]
        if self.is_testing or self.is_debug:
            print('Fetch {}th datapoint from {}'.format(i, path))
        # name = os.path.splitext(os.path.basename(path))[0]
        item = path.split('.')[0].split('/')[-2].split('_')[0]
        norm_factor_instance = self.all_factors[item]
        corner_pts_instance  = self.all_corners[item]
        joints = self.all_joints[item]
        if self.is_debug:
            print('Now fetching {}th data from instance {} with norm_factors: {}'.format(i, item, norm_factor_instance))
        with h5py.File(path, 'r') as handle:
            data = self.create_unit_data_from_hdf5(handle, self.n_max_parts, self.num_points, parts_map=self.parts_map, instance=item,  \
                                norm_factors=norm_factor_instance, norm_corners=corner_pts_instance, joints=joints, nocs_type=self.nocs_type, \
                                add_noise=self.add_noise, fixed_order=self.fixed_order, shuffle=not self.fixed_order, line_space=self.line_space,\
                                is_testing=self.is_testing)
  
        if self.is_testing or self.is_debug:
            return data, path
        return data

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
            # open a file, where you stored the pickled data
            file = open(root_dset + "/pickle/{}.pkl".format(obj_category),"rb")
            # dump information to that file
            data = pickle.load(file)
            all_factors = data
            file.close()
            fc = open(root_dset + "/pickle/{}_corners.pkl".format(obj_category), "rb")
            all_corners = pickle.load(fc)
            fc.close()
        if is_debug:
            print('Now fetching nocs normalization factors', type(all_factors), all_factors)

        return all_factors, all_corners

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

    def create_unit_data_from_hdf5(self, f, n_max_parts, num_points, parts_map = [[0, 3, 4], [1, 2]], instance=None, \
                                norm_corners=[None], norm_factors=[None], joints=None, nocs_type='A', line_space='orth', thres_r=0.2,\
                                add_noise=False, fixed_order=False, check_only=False, shuffle=True,\
                                is_testing=False, is_debug=False):
        n_parts   = len(parts_map)
        nocs_p, nocs_g, nocs_n, parts_cls, parts_pts, offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params, hand_joints_heatmap, hand_joints_unitvec, hand_joints_cls, n_total_points = self.create_data_shape2motion(f, n_max_parts, num_points, parts_map = parts_map, instance=instance,\
                                    norm_corners=norm_corners, norm_factors=norm_factors, joints=joints, nocs_type=nocs_type, line_space=line_space, thres_r=thres_r,\
                                    add_noise=add_noise, fixed_order=fixed_order, check_only=check_only, shuffle=shuffle, \
                                    is_testing=is_testing, is_debug=is_debug)
        # get hand heatmap
        J_num = 21
        hand_joints_offset_heatmap = np.zeros([n_total_points, 1*J_num], dtype=np.float32)
        hand_joints_offset_unitvec = np.zeros([n_total_points, 3*J_num], dtype=np.float32)
        hand_joint_cls  = np.zeros([n_total_points], dtype=np.float32)

        hand_joints_offset_heatmap[-parts_pts[-1].shape[0]:] = hand_joints_heatmap
        hand_joints_offset_unitvec[-parts_pts[-1].shape[0]:] = hand_joints_unitvec
        hand_joint_cls[-parts_pts[-1].shape[0]:] = 1 # hand points 
        # add 
        cls_arr = np.concatenate(parts_cls, axis=0)
        pts_arr = np.concatenate(parts_pts, axis=0)

        # you may want to add zeros points
        offset_heatmap =  np.concatenate(offset_heatmap, axis=0)
        if is_debug:
            print('offset_heatmap max is {}'.format(np.amax(offset_heatmap)))
        offset_unitvec =  np.concatenate(offset_unitvec, axis=0)
        joint_orient   =  np.concatenate(joint_orient, axis=0)
        joint_cls      =  np.concatenate(joint_cls, axis=0)

        if nocs_p[0] is not None:
            p_arr = np.concatenate(nocs_p, axis=0)
        if nocs_n[0] is not None:
            n_arr = np.concatenate(nocs_n, axis=0)
        if nocs_g[0] is not None:
            g_arr = np.concatenate(nocs_g, axis=0)

        if n_parts > n_max_parts:
            print('n_parts {} > n_max_parts {}'.format(n_parts, n_max_parts))
            return None

        if np.amax(cls_arr) >= n_parts:
            print('max label {} > n_parts {}'.format(np.amax(cls_arr), n_parts))
            return None

        if n_total_points < num_points:
            # print('tiling points, n_total_points {} < num_points {} required'.format(n_total_points, num_points))
            # we'll tile the points
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
            hnad_joint_cls_tiled  = np.concatenate([joint_cls] * tile_n, axis=0)
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

        if is_testing: # return the original unsmapled data
            if self.name_dset == 'sapien':
                target_order = self.ctgy_spec.spec_map[instance]
                joint_rpy = joints['joint']['rpy'][target_order[0]]
                rot_mat = euler_matrix(joint_rpy[0], joint_rpy[1], joint_rpy[2])[:3, :3]
                p_arr   = np.dot(p_arr-0.5, rot_mat.T) + 0.5
                g_arr   = np.dot(g_arr-0.5, rot_mat.T) + 0.5
            result = {
                'P': pts_arr*norm_factors[0], # todo
                'cls_gt': cls_arr.astype(np.float32),
                'mask_array': mask_array.astype(np.float32),
                'nocs_gt': p_arr,
                'nocs_gt_g': g_arr,
            }
            return result

        perm          = np.random.permutation(n_total_points)
        cls_arr       = cls_arr[perm[:num_points]]
        if self.name_dset == 'BMVC15':
            pts_arr       = pts_arr[perm[:num_points]]
        else:
            pts_arr       = pts_arr[perm[:num_points]] * norm_factors[0]
        offset_heatmap_arr = offset_heatmap[perm[:num_points]]
        offset_unitvec_arr = offset_unitvec[perm[:num_points]]
        joint_orient_arr   =  joint_orient[perm[:num_points]]
        joint_cls_arr  = joint_cls[perm[:num_points]]
        # print('joint_cls_arr has shape: ', joint_cls_arr.shape)
        joint_cls_mask = np.zeros((joint_cls_arr.shape[0]), dtype=np.float32)
        id_valid     = np.where(joint_cls_arr>0)[0]
        joint_cls_mask[id_valid] = 1.00
        hand_joints_offset_heatmap = hand_joints_offset_heatmap[perm[:num_points]]
        hand_joints_offset_unitvec = hand_joints_offset_unitvec[perm[:num_points]]
        hand_joint_cls = hand_joint_cls[perm[:num_points]]
        
        # 
        mask_array[np.arange(num_points), cls_arr.astype(np.int8)] = 1.00
        mask_array[np.arange(num_points), n_parts-1] = 0.0 # ignore hand points for nocs prediction

        if nocs_p[0] is not None:
            p_arr = p_arr[perm[:num_points]]
        if nocs_n[0] is not None:
            n_arr = n_arr[perm[:num_points]]
        if nocs_g[0] is not None:
            g_arr = g_arr[perm[:num_points]]

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

        if nocs_type == 'A':
            result = {
                'P': pts_arr,
                'cls_gt': cls_arr.astype(np.float32),
                'mask_array': mask_array.astype(np.float32),
                'nocs_gt'   : p_arr.astype(np.float32),
                'nocs_gt_g' : g_arr.astype(np.float32),
                'hand_heatmap_gt'   : hand_joints_offset_heatmap.astype(np.float32),
                'hand_unitvec_gt'   : hand_joints_offset_unitvec.astype(np.float32),
                'hand_joint_cls_mask': hand_joint_cls.astype(np.float32),
                'heatmap_gt'   : offset_heatmap_arr.astype(np.float32),
                'unitvec_gt'   : offset_unitvec_arr.astype(np.float32),
                'orient_gt'    : joint_orient_arr.astype(np.float32),
                'joint_cls_gt'    : joint_cls_arr.astype(np.float32),
                'joint_cls_mask'  : joint_cls_mask.astype(np.float32),
                'joint_params_gt' : joint_params,
            }

        elif nocs_type == 'AC':
            result = {
                'P': pts_arr,
                'cls_gt': cls_arr.astype(np.float32),
                'mask_array': mask_array.astype(np.float32),
                'nocs_gt'   : p_arr.astype(np.float32),
                'nocs_gt_g' : g_arr.astype(np.float32),
                'hand_heatmap_gt'   : hand_joints_offset_heatmap.astype(np.float32),
                'hand_unitvec_gt'   : hand_joints_offset_unitvec.astype(np.float32),
                'hand_joint_cls_mask': hand_joint_cls.astype(np.float32),
                'heatmap_gt'   : offset_heatmap_arr.astype(np.float32),
                'unitvec_gt'   : offset_unitvec_arr.astype(np.float32),
                'orient_gt'    : joint_orient_arr.astype(np.float32),
                'joint_cls_gt'    : joint_cls_arr.astype(np.float32),
                'joint_cls_mask'  : joint_cls_mask.astype(np.float32),
                'joint_params_gt' : joint_params,
            }

        return result

    def create_data_shape2motion(self, f, n_max_parts, num_points, parts_map = [[0, 3, 4], [1, 2]], instance=None, \
                                norm_corners=[None], norm_factors=[None], joints=None, nocs_type='A', line_space='orth', thres_r=0.2,\
                                add_noise=False, fixed_order=False, check_only=False, shuffle=True, \
                                is_testing=False, is_debug=False):
        '''
            f will be a h5py group-like object
        '''
        # read
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

        joint_xyz = joints['link']['xyz']
        joint_rpy = joints['joint']['axis']
        joint_part= joints['joint']['parent']
        joint_type= joints['joint']['type']

        joint_params = np.zeros((n_parts, 7))
        if line_space == 'plucker':
            joint_params = np.zeros((n_parts, 6))

        # get joints in camera space 
        hand_joints = f['joints'][()]
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

        # breakpoint()
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
                        # if is_debug:
                        #     plot_arrows(nocs_g[j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, m))
        # get hand_joints_offsets, todo
        J_num = 21 
        # hand_joints_offsets = hand_joints_offsets.reshape(-1, J_num*3) # todo
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

        # breakpoint()
        for j, offsets in enumerate(parts_offset_joint):
            offset_heatmap[j] = np.zeros((parts_gts[j].shape[0]))
            offset_unitvec[j] = np.zeros((parts_gts[j].shape[0], 3))
            joint_orient[j]   = np.zeros((parts_gts[j].shape[0], 3))
            joint_cls[j]      = np.zeros((parts_gts[j].shape[0]))

            # by default it is zero
            for k, offset in enumerate(offsets):
                heatmap = np.linalg.norm(offset, axis=1)
                unitvec = offset/(heatmap.reshape(-1, 1) + epsilon)
                idc     = np.where(heatmap<thres_r)[0]
                offset_heatmap[j][idc]    = 1 - heatmap[idc]/thres_r

                offset_unitvec[j][idc, :] = unitvec[idc, :]
                joint_orient[j][idc, :]   = parts_joints[j][k][1]
                joint_cls[j][idc]         = joint_index[j][k]

        if nocs_type == 'C':
            if is_debug:
                plot_arrows_list(nocs_g, parts_offset_joint, parts_joints, title_name='joint offset')
                plot_arrows_list_threshold(nocs_g, parts_offset_joint, parts_joints, title_name='joint offset')

        return  nocs_p, nocs_g, nocs_n, parts_cls, parts_pts, offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params, hand_joints_heatmap, hand_joints_unitvec, hand_joints_cls, n_total_points
        

@hydra.main(config_path="../config/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(cfg.port)
    infos       = global_info()
    data_infos  = infos.datasets[cfg.item]

    # category-wise training setup
    cfg.n_max_parts = data_infos.num_parts
    cfg.is_test = False

    random.seed(30)

    train_data = SyntheticDataset(
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
        mode='train',
        fixed_order=False,)

    np.random.seed(0)
    selected_index = np.random.randint(len(train_data.basename_list), size=2)
    # selected_index = [train_data.basename_list.index('0016_0_0')] + list(np.arange(0, len(train_data.basename_list)))
    for i in selected_index:
        basename =  train_data.basename_list[i]
        if basename.split('_')[0] not in test_ins:
            continue
        instance = basename.split('_')[0]
        print('reading data point: ', i, train_data.basename_list[i])
        if is_debug or is_testing:
            data_pts, _ =  train_data.fetch_data_at_index(i)
        else:
            data_pts=  train_data.fetch_data_at_index(i)
        # print('fetching ', path)
        for keys, item in data_pts.items():
            print(keys, item.shape)

        input_pts = data_pts['P']
        nocs_gt   = {}
        nocs_gt['pn']   = data_pts['nocs_gt']
        if args.nocs_type == 'AC':
            nocs_gt['gn']   = data_pts['nocs_gt_g']
        # print('nocs_gt has {}, {}'.format( np.amin(nocs_gt['pn'], axis=0), np.amax(nocs_gt, axis=0)))
        mask_gt   = data_pts['cls_gt']
        num_pts = input_pts.shape[0]
        num_parts = n_max_parts
        part_idx_list_gt   = []
        for j in range(num_parts):
            part_idx_list_gt.append(np.where(mask_gt==j)[0])
        if not args.test:
            heatmap_gt= data_pts['heatmap_gt']
            unitvec_gt= data_pts['unitvec_gt']
            orient_gt = data_pts['orient_gt']
            joint_cls_gt = data_pts['joint_cls_gt']
            joint_params_gt = data_pts['joint_params_gt']
            # print('joint_params_gt is: ', joint_params_gt)
            joint_idx_list_gt   = []
            for j in range(num_parts):
                joint_idx_list_gt.append(np.where(joint_cls_gt==j)[0])

        #>>>>>>>>>>>>>>>>>>>>>------ For segmentation visualization ----
        # plot_imgs([rgb_img], ['rgb img'], title_name='RGB', sub_name=str(i), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([[input_pts]], [['Part {}'.format(0)]], s=50, title_name=['GT seg on input point cloud'], sub_name=str(i), show_fig=args.show_fig, axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([[input_pts[part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=50, title_name=['GT seg on input point cloud'], sub_name=str(i), show_fig=args.show_fig, axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([[nocs_gt['gn'][part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=50, title_name=['GT global NOCS'], sub_name=str(i), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([[nocs_gt['pn'][part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=50, title_name=['GT part NOCS'], sub_name=str(i), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        for j in range(num_parts):
            plot3d_pts([[nocs_gt['pn'][part_idx_list_gt[j], :] ]], [['Part {}'.format(j)]], s=15, dpi=200, title_name=['GT part NOCS'], color_channel=[[ nocs_gt['pn'][part_idx_list_gt[j], :] ]], sub_name='{}_part_{}'.format(i, j) , show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
            plot3d_pts([[nocs_gt['gn'][part_idx_list_gt[j], :] ]], [['Part {}'.format(j)]], s=15, dpi=200, title_name=['GT global NOCS'], color_channel=[[ nocs_gt['gn'][part_idx_list_gt[j], :] ]], sub_name='{}_part_{}'.format(i, j), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)

        # show joints
        if not cfg.is_test:
            plot3d_pts([[input_pts[joint_idx_list_gt[j], :] for j in range(num_parts)]], [['unassigned Pts'] + ['Pts of joint {}'.format(j) for j in range(1, num_parts)]], s=15, \
                  title_name=['GT association of pts to joints '], sub_name=str(i), show_fig=args.show_fig, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item, axis_off=True)
            plot3d_pts([[ input_pts ]], [['Part 0-{}'.format(num_parts-1)]], s=15, \
                  dpi=200, title_name=['Input Points distance heatmap'], color_channel=[[250*np.concatenate([heatmap_gt.reshape(-1, 1), np.zeros((heatmap_gt.shape[0], 2))], axis=1)]], show_fig=args.show_fig)

            thres_r       = 0.2
            offset        = unitvec_gt * (1- heatmap_gt.reshape(-1, 1)) * thres_r
            joint_pts     = nocs_gt['gn'] + offset
            joints_list   = []
            idx           = np.where(joint_cls_gt > 0)[0]
            plot_arrows(nocs_gt['gn'][idx], [0.5*orient_gt[idx]], whole_pts=nocs_gt['gn'], title_name='{}_joint_pts_axis'.format('GT'), dpi=200, s=25, thres_r=0.1, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
            plot_arrows(nocs_gt['gn'][idx], [offset[idx]], whole_pts=nocs_gt['gn'], title_name='{}_joint_offsets'.format('GT'), dpi=200, s=25, thres_r=0.1, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
            # plot_arrows(input_pts[idx], [0.5*orient_gt[idx]], whole_pts=input_pts, title_name='{}_joint_pts_axis'.format('GT'), dpi=200, s=25, thres_r=0.2, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
            # plot_arrows(input_pts[idx], [offset[idx]], whole_pts=input_pts, title_name='{}_joint_offsets'.format('GT'), dpi=200, s=25, thres_r=0.2, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)

if __name__=='__main__':
    main()