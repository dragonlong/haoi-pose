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
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf

import __init__
from global_info import global_info
from dataset.base import BaseDataset
from dataset.parser import Parser
epsilon = 10e-8
MAX_NUM_OBJ = 64

def breakpoint():
    import pdb;pdb.set_trace()


class OccupancyDataset(BaseDataset):
    def __init__(self, cfg, mode='train', domain=None, first_n=-1, add_noise=False, fixed_order=False, num_expr=0.01):
        BaseDataset.__init__(self, cfg=cfg, mode=mode, domain=domain, first_n=first_n, add_noise=add_noise, fixed_order=fixed_order, num_expr=num_expr)

    def __getitem__(self, i):
        assert not self.first_iteration_finished
        thres_r = 0.2
        n_parts = self.n_max_parts
        path = self.hdf5_file_list[i]
        if self.is_testing or self.is_debug:
            print('Fetch {}th datapoint from {}'.format(i, path))

        with h5py.File(path, 'r') as handle:
            base_name = self.basename_list[i]
            mean_hand, hand_joints, hand_contacts, parts_pts, parts_cls, nocs_p, nocs_g, n_total_points = self.get_h5_data(handle, base_name=base_name)
            # hand_joints_heatmap, hand_joints_unitvec, hand_joints_cls = self.get_hand_offsets(hand_joints, hand_contacts, parts_pts, thres_r=0.2)
            # offset_heatmap, offset_unitvec, joint_orient, joint_cls, joint_params = self.get_object_offsets(nocs_g, base_name, thres_r=0.2)

        # Observation

        # GT occupancy for sampled points

        # >>>>>>>>>>>>>>>>>>>> points & seg labels
        pts_arr = np.concatenate(parts_pts, axis=0)
        cls_arr = np.concatenate(parts_cls, axis=0)
        p_arr   = np.concatenate(nocs_p, axis=0)
        g_arr   = np.concatenate(nocs_g, axis=0)
        # put into a list
        output_arr = [pts_arr, cls_arr, p_arr, g_arr]
        if n_total_points < self.num_points:
            tile_n = int(self.num_points/n_total_points) + 1
            n_total_points = tile_n * n_total_points
            for j in range(len(output_arr)):
                arr_tiled = np.concatenate([np.copy(output_arr[j])] * tile_n, axis=0)
                output_arr[j] = arr_tiled

        # use furthest point sampling
        perm       = np.random.permutation(n_total_points)
        sample_ind = np.arange(0, self.num_points).tolist()
        for j in range(len(output_arr)):
            output_arr[j] = output_arr[j][perm][sample_ind]

        # assign them back
        pts_arr, cls_arr, p_arr, g_arr = output_arr
        mask_array    = np.zeros([self.num_points, n_parts], dtype=np.float32)
        mask_array[np.arange(self.num_points), cls_arr.astype(np.int8)] = 1.00
        mask_array[np.arange(self.num_points), n_parts-1] = 0.0 # ignore hand points for nocs prediction

        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        target_bboxes_mask[0:hand_contacts.shape[0]] = 1
        target_bboxes[0:hand_contacts.shape[0],:3] = hand_contacts[:,0:3]

        # loop contacts
        # print('hand_contacts: ', hand_contacts)
        point_votes = hand_contacts[np.newaxis, :, :] - pts_arr[:, np.newaxis, :]# N, 1, 3 - 1, n, 3
        point_votes_distance = np.linalg.norm(point_votes, axis=2)

        # find minimal,
        pt2ct_ind      = np.argmin(point_votes_distance, axis=1)
        point_votes    = point_votes[np.arange(self.num_points), pt2ct_ind, :]

        point_votes_distance = point_votes_distance[np.arange(self.num_points), pt2ct_ind].reshape(-1)

        # set threshold, objectness mask
        point_votes_mask = np.zeros((self.num_points))
        point_votes_mask[np.where(point_votes_distance<thres_r)[0]] = 1.0

        point_votes    = point_votes * point_votes_mask[:, np.newaxis]
        point_votes    = np.tile(point_votes, (1, 3)) # make 3 votes identical
        mask_array_hand = np.zeros([self.num_points, 2], dtype=np.float32)
        hand_ind = np.where(cls_arr>2)[0]
        obj_ind  = np.where(cls_arr<3)[0]
        mask_array_hand[hand_ind, 1]= 1.0
        mask_array_hand[obj_ind, 0]= 1.0
        pts_arr = np.concatenate([pts_arr, mask_array_hand], axis=1)
        #  & offsets voting
        if not self.is_testing:
            pts_arr = torch.from_numpy(pts_arr.astype(np.float32))
            cls_arr = torch.from_numpy(cls_arr.astype(np.float32))
            mask_array = torch.from_numpy(mask_array.astype(np.float32)) #
            p_arr = torch.from_numpy(p_arr.astype(np.float32)) # .transpose(1, 0)
            g_arr = torch.from_numpy(g_arr.astype(np.float32)) # .transpose(1, 0)
            hand_joints = torch.from_numpy(hand_joints.astype(np.float32))
            hand_contacts = torch.from_numpy(hand_contacts.astype(np.float32))
            point_votes   = torch.from_numpy(point_votes.astype(np.float32))
            point_votes_mask = torch.from_numpy(point_votes_mask.astype(np.int64))
            # hand_poses  = torch.from_numpy(hand_poses.astype(np.float32))
            # hand_trans  = torch.from_numpy(hand_trans.astype(np.float32))

            # hand_tf = torch.from_numpy(hand_tf[:3, :3].astype(np.float32)) # original hand poses
            # RT_full = torch.from_numpy(RT_full.astype(np.float32))
            # mean_hand = torch.from_numpy(mean_hand.astype(np.float32))
            # hand_rotation = torch.from_numpy(hand_rotation.astype(np.float32))
        data = {
            'P': pts_arr,
            'partcls_per_point': cls_arr, #
            'part_mask': mask_array,
            'nocs_per_point' : p_arr,
            'gocs_per_point' : g_arr,
            'vote_label' : point_votes,
            'vote_label_mask': point_votes_mask,
            'center_label': torch.from_numpy(target_bboxes.astype(np.float32)[:,0:3]),
            'box_label_mask':  torch.from_numpy(target_bboxes_mask.astype(np.float32)),
            # 'extrinsic_params': RT_full,
            # 'hand_contacts': hand_contacts,
            # 'mean_hand': mean_hand,
            # 'original_poses': hand_tf,
        }
        data['index'] = i

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
    # dset   = OccupancyDataset(cfg, mode='train')
    # dset.__getitem__(0)
    parser = Parser(cfg, OccupancyDataset) # we could choose the Dataset to pass into here



if __name__ == '__main__':
    main()
