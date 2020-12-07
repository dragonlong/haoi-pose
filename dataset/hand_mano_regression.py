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
import scipy
import scipy.io
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf
from multiprocessing import Manager
from pytransform3d.rotations import *

import __init__
from global_info import global_info
from dataset.base import BaseDataset
from dataset.parser import Parser
from common.d3_utils import point_3d_offset_joint, mat_from_rvec, rvec_from_mat
epsilon = 10e-8
MAX_NUM_OBJ = 64

def breakpoint():
    import pdb;pdb.set_trace()


class ManoRegressionDataset(BaseDataset):
    def __init__(self, cfg, mode='train', domain=None, first_n=-1, add_noise=False, fixed_order=False, num_expr=0.01):
        BaseDataset.__init__(self, cfg=cfg, mode=mode, domain=domain, first_n=first_n, add_noise=add_noise, fixed_order=fixed_order, num_expr=num_expr)

    def __getitem__(self, i):
        assert not self.first_iteration_finished
        n_parts = self.n_max_parts
        path = self.hdf5_file_list[i]
        if self.is_testing or self.is_debug:
            print('Fetch {}th datapoint from {}'.format(i, path))
        with h5py.File(path, 'r') as handle:
            base_name = self.basename_list[i]
            mean_hand, hand_joints, hand_contacts, parts_pts, parts_cls, nocs_p, nocs_g, n_total_points = self.get_h5_data(handle, base_name=base_name)
            hand_joints_heatmap, hand_joints_unitvec, hand_joints_cls = self.get_hand_offsets(hand_joints, hand_contacts, parts_pts, thres_r=0.2)
            offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params = self.get_object_offsets(nocs_g, base_name, thres_r=0.2)

        # >>>>>>>>>>>>>>>>>>>>> hand poses
        hand_poses = self.mano_params["_".join(["{:}".format(x) for x in base_name.split('_')[:-1]])].reshape(-1) # normalize to 1
        hand_trans = self.mano_trans["_".join(["{:}".format(x) for x in base_name.split('_')[:-1]])]
        hand_tf     = mat_from_rvec(hand_poses[:3]) # original hand pose
        # camera R, T, extrinsic
        frame_ind = base_name.split('_')[-1]
        meta_file = '/groups/CESCA-CV/ICML2021/data/render/eyeglasses/' + base_name[:-(len(frame_ind)+1)] + '/meta_{:04d}.mat'.format(int(frame_ind))
        meta = scipy.io.loadmat(meta_file)
        RT_full = meta['rotation_translation_matrix']

        #>>>>>>>>>>> core code
        rot2blender = matrix_from_euler_xyz([-np.pi/2, 0, 0])
        rot_canon   = mat_from_rvec(hand_poses[:3])
        rot2camera  = RT_full[:3, :3]
        trans2camera= RT_full[:3, 3]

        #>>>>>>>>>>>>> used for network input
        hand_rotation = np.asarray(rot2camera @ rot2blender @ rot_canon)
        if self.is_testing:
            print('--- not testing!!! use hand poses in camera space')
        else:
            # use hand poses in camera space
            rot_in_6d   = hand_rotation[:, :2].T.reshape(1, -1)
            hand_poses  = np.asarray(np.concatenate([rot_in_6d, hand_poses[3:].reshape(1, -1)], axis=1)).reshape(-1)#
            assert hand_poses.shape[0] == 51, print('wrong hand_poses dimension: ', hand_poses.shape[0])
        # scale = 200
        # hand_trans  = rot2camera @ rot2blender @ hand_trans.reshape(3, 1) * 1000 / scale + trans2camera.reshape(3, 1)
        # hand_trans  = (hand_trans / (1000/scale)).reshape(1, 3)
        # hand_trans  = hand_trans - mean_hand
        #>>>>>>>>>>>>>>>>> core code ends here <<<<<<<<<<<<<<<<<<<<<<<<<#

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
        offset_unitvec =  np.concatenate(offset_unitvec, axis=0)
        joint_orient   =  np.concatenate(joint_orient, axis=0)
        joint_cls      =  np.concatenate(joint_cls, axis=0)

        #>>>>>>>>>>>>>>>>>>>>> NOCS value
        p_arr = np.concatenate(nocs_p, axis=0)
        g_arr = np.concatenate(nocs_g, axis=0)

        if np.amax(cls_arr) >= n_parts:
            print('max label {} > n_parts {}'.format(np.amax(cls_arr), n_parts))
            return None

        output_arr = [pts_arr, cls_arr, p_arr, g_arr, offset_heatmap, offset_unitvec, joint_orient, joint_cls, hand_joints_offset_heatmap, hand_joints_offset_unitvec, hand_joint_cls]
        if self.hand_only and np.where(cls_arr==self.parts_map[-1])[0].shape[0] > 0:
            n_total_points   = np.where(cls_arr==self.parts_map[-1])[0].shape[0]

        if n_total_points < self.num_points:
            tile_n = int(self.num_points/n_total_points) + 1
            n_total_points = tile_n * n_total_points
            for j in range(len(output_arr)):
                arr_tiled = np.concatenate([np.copy(output_arr[j])] * tile_n, axis=0)
                output_arr[j] = arr_tiled

        pts_arr, cls_arr, p_arr, g_arr, offset_heatmap, offset_unitvec, joint_orient, joint_cls, hand_joints_offset_heatmap, hand_joints_offset_unitvec, hand_joint_cls= output_arr
        mask_array    = np.zeros([self.num_points, n_parts], dtype=np.float32)

        perm       = np.random.permutation(n_total_points)
        if self.hand_only and np.where(cls_arr[perm]==self.parts_map[-1])[0].shape[0] > 2047:
            hand_inds  = np.where(cls_arr[perm]==self.parts_map[-1])[0]
            sample_ind = hand_inds[:self.num_points].tolist()
            assert len(sample_ind) == self.num_points, 'wrong shape, {}'.format(len(sample_ind))
        else:
            sample_ind = np.arange(0, self.num_points).tolist()
        # sample_ind    = farthest_point_sample(self.num_points, pts_arr[perm][np.newaxis, :, :])
        # sample_ind = sample_ind.eval()[0, :]

        pts_arr       = pts_arr[perm][sample_ind] #* norm_factors[0] # by default is 1
        cls_arr       = cls_arr[perm][sample_ind]
        # mask_array[np.arange(self.num_points), cls_arr.astype(np.int16)] = np.ones((self.num_points))
        # mask_array[np.arange(self.num_points), n_parts-1] = 0.0 # ignore hand points for nocs prediction

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

        p_arr = p_arr[perm][sample_ind]
        g_arr = g_arr[perm][sample_ind]

        if not self.is_testing:
            pts_arr = torch.from_numpy(pts_arr.astype(np.float32).transpose(1, 0))
            cls_arr = torch.from_numpy(cls_arr.astype(np.float32))
            mask_array = torch.from_numpy(mask_array.astype(np.float32).transpose(1, 0))
            p_arr = torch.from_numpy(p_arr.astype(np.float32).transpose(1, 0))
            g_arr = torch.from_numpy(g_arr.astype(np.float32).transpose(1, 0))
            hand_joints = torch.from_numpy(hand_joints.astype(np.float32))
            hand_contacts = torch.from_numpy(hand_contacts.astype(np.float32))
            hand_poses  = torch.from_numpy(hand_poses.astype(np.float32))
            hand_trans  = torch.from_numpy(hand_trans.astype(np.float32))

            hand_joints_offset_heatmap = torch.from_numpy(hand_joints_offset_heatmap.astype(np.float32).transpose(1, 0))
            hand_joints_offset_unitvec = torch.from_numpy(hand_joints_offset_unitvec.astype(np.float32).transpose(1, 0))
            hand_joint_cls = torch.from_numpy(hand_joint_cls.astype(np.float32))

            offset_heatmap_arr = torch.from_numpy(offset_heatmap_arr.astype(np.float32))
            offset_unitvec_arr = torch.from_numpy(offset_unitvec_arr.astype(np.float32).transpose(1, 0))
            joint_orient_arr = torch.from_numpy(joint_orient_arr.astype(np.float32).transpose(1, 0))
            joint_cls_arr = torch.from_numpy(joint_cls_arr.astype(np.float32))
            joint_cls_mask = torch.from_numpy(joint_cls_mask.astype(np.float32))
            joint_params = torch.from_numpy(joint_params)
            hand_tf = torch.from_numpy(hand_tf[:3, :3].astype(np.float32)) # original hand poses
            RT_full = torch.from_numpy(RT_full.astype(np.float32))
            mean_hand = torch.from_numpy(mean_hand.astype(np.float32))
            hand_rotation = torch.from_numpy(hand_rotation.astype(np.float32))

        if self.pred_mano:
            regression_params = hand_poses
        else:
            regression_params = hand_joints

        data = {
            'P': pts_arr,
            'partcls_per_point': cls_arr, #
            'regressionR': hand_rotation, # not hand_poses
            'regression_params': regression_params, # 21, 3
            'hand_mask' : hand_joint_cls,
            'extrinsic_params': RT_full, # 3, 4
            # 'hand_contacts': hand_contacts, #
            'hand_joints': hand_joints,
            'mean_hand': mean_hand,
            'original_poses': hand_tf,
        }
        data['index'] = i

        assert data['P'].shape[1] == 2048, 'wrong pts_arr'
        assert data['partcls_per_point'].shape[0] == 2048, 'wrong partcls_per_point'
        assert data['regressionR'].shape[1] == 3, 'wrong regressionR'
        assert data['regression_params'].shape[0] == 51, 'wrong regression_params'
        assert data['hand_mask'].shape[0] == 2048, print('wrong hand_mask')
        assert data['extrinsic_params'].shape[1] == 4, 'wrong extrinsic_params'
        # assert data['hand_contacts'].shape[1] == 3, print('wrong hand_contacts')
        assert data['mean_hand'].shape[1] == 3, 'wrong mean_hand'
        assert data['original_poses'].shape[1] == 3, 'wrong original_poses'

        return data

@hydra.main(config_path="../config/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(cfg.port)
    infos           = global_info()
    data_infos      = infos.datasets[cfg.item]
    base_path       = infos.base_path
    group_dir       = infos.group_path
    second_path     = infos.second_path

    # category-wise training setup
    cfg.n_max_parts = data_infos.num_parts
    cfg.is_test = False

    random.seed(30)
    parser = Parser(cfg, ManoRegressionDataset) # we could choose the Dataset to pass into here
    for j in range(10):

        data = parser.train_dataset.__getitem__(0)
        for key, value in data.items():
            try:
                print(key, value.shape)
            except:
                print(key, value)
if __name__ == '__main__':
    main()
