#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling SemanticKitti dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#      Adpated by Xiaolong Li
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#
# Common libs
import time
import numpy as np
import pickle
import torch
import yaml
from multiprocessing import Lock
from random import randint, sample

# OS functions
from os import listdir
from os.path import exists, join, isdir
from torch.utils.data import Sampler, get_worker_info

# Dataset parent class
import __init__
from dataset.kitti.common import *

from common.mayavi_visu import *
from common.metrics import fast_confusion

from dataset.kitti.common import grid_subsampling
from common.vis_utils import bcolors

def breakpoint():
    import pdb;pdb.set_trace()

# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/

class SemanticKittiDataset(PointCloudDataset):
    """Class to handle SemanticKitti dataset."""

    def __init__(self, config, set='training', balance_classes=True, fixed_centers=False):
        PointCloudDataset.__init__(self, 'SemanticKitti', config)

        ##########################
        # Parameters for the files
        ##########################
        # Dataset folder
        # self.path = '/groups/CESCA-CV/dataset'
        # self.path = '/data2/datasets/tmp/kitti/dataset'
        self.path = config.root_dataset

        # Type of task conducted on this dataset
        self.dataset_task = 'slam_segmentation'

        # Training or test set
        self.set = set

        # Get a list of sequences
        if self.set == 'training':
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i != 8]
        elif self.set == 'validation':
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 8]
        elif self.set == 'test':
            self.sequences = ['{:02d}'.format(i) for i in range(11, 22)]
        else:
            raise ValueError('Unknown set for SemanticKitti data: ', self.set)

        # List all files in each sequence
        self.frames = []
        for seq in self.sequences:
            velo_path = join(self.path, 'sequences', seq, 'velodyne')
            frames = np.sort([vf[:-4] for vf in listdir(velo_path) if vf.endswith('.bin')])
            if self.set == 'validation' and config.n_frames > 1: # TODO, iff
                self.frames.append(frames[131:])
            else:
                self.frames.append(frames)

        ###########################
        # Object classes parameters
        ###########################

        # Read labels
        self.nframe = config.n_frames
        config_file = join(self.path, 'semantic-kitti-all.yaml') # only train it for 20 classes TODO

        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            learning_map_inv = doc['learning_map_inv']
            learning_map = doc['learning_map']
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

            self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v

        # Dict from labels to names
        self.label_to_names = {k: all_labels[v] for k, v in learning_map_inv.items()}
        self.label_to_motion  = {0: 'background', 1:'static', 2: 'moving'}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])
        self.motion_map = {
                  0 : 0,     # "unlabeled"
                  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
                  10: 1,     # "car"
                  11: 1,     # "bicycle"
                  13: 1,     # "bus" mapped to "other-vehicle" --------------------------mapped
                  15: 1,     # "motorcycle"
                  16: 1,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
                  18: 1,     # "truck"
                  20: 1,     # "other-vehicle"
                  30: 1,     # "person"
                  31: 1,     # "bicyclist"
                  32: 1,     # "motorcyclist"
                  40: 0,     # "road"
                  44: 0,    # "parking"
                  48: 0,    # "sidewalk"
                  49: 0,    # "other-ground"
                  50: 0,    # "building"
                  51: 0,    # "fence"
                  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
                  60: 0,     # "lane-marking" to "road" ---------------------------------mapped
                  70: 0,    # "vegetation"
                  71: 0,    # "trunk"
                  72: 0,    # "terrain"
                  80: 0,    # "pole"
                  81: 0,    # "traffic-sign"
                  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
                  252: 2,    # "moving-bicyclist"
                  254: 2,    # "moving-person"
                  255: 2,    # "moving-motorcyclist"
                  256: 2,    # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
                  257: 2,    # "moving-bus" mapped to "moving-other-vehicle" -----------mapped
                  258: 2,    # "moving-truck"
                  259: 2,}    # "moving-other-vehicle"}
        self.motion_remap = {
                  0: 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
                  1: 1,    # "car"
                  2: 1,    # "bicycle"
                  3: 1,    # "motorcycle"
                  4: 1,    # "truck"
                  5: 1,    # "on-rails" mapped to "other-vehicle" ---------------------mapped
                  6: 1,    # "person"
                  7: 1,    # "bicyclist"
                  8: 1,    # "motorcyclist"
                  9: 0,    # "road"
                  10: 0,   # "parking"
                  11: 0,   # "sidewalk"
                  12: 0,   # "other-ground"
                  13: 0,   # "building"
                  14: 0,   # "fence"
                  15: 0,   # "vegetation"
                  16: 0,   # "trunk"
                  17: 0,   # "terrain"
                  18: 0,   # "pole"
                  19: 0,   # "traffic-sign"
                  20: 2,   # "moving-car"
                  21: 2,   # "moving-bicyclist"
                  22: 2,   # "moving-person"
                  23: 2,   # "moving-motorcyclist"
                  24: 2,   # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
                  25: 2,   # "moving-truck"
                          }
        self.semantic_remap = {
                  0: 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
                  1: 1,    # "car"
                  2: 2,    # "bicycle"
                  3: 3,    # "motorcycle"
                  4: 4,    # "truck"
                  5: 5,    # "on-rails" mapped to "other-vehicle" ---------------------mapped
                  6: 6,    # "person"
                  7: 7,    # "bicyclist"
                  8: 8,    # "motorcyclist"
                  9: 9,    # "road"
                  10: 10,   # "parking"
                  11: 11,   # "sidewalk"
                  12: 12,   # "other-ground"
                  13: 13,   # "building"
                  14: 14,   # "fence"
                  15: 15,   # "vegetation"
                  16: 16,   # "trunk"
                  17: 17,   # "terrain"
                  18: 18,   # "pole"
                  19: 19,   # "traffic-sign"
                  20: 1,   # "moving-car"
                  21: 7,   # "moving-bicyclist"
                  22: 6,   # "moving-person"
                  23: 8,   # "moving-motorcyclist"
                  24: 5,   # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
                  25: 16,   # "moving-truck"
                          }
        self.motion_lut = np.zeros((np.max([k for k in self.motion_remap.keys()]) + 1), dtype=np.int32)
        for k, v in self.motion_remap.items():
            self.motion_lut[k] = v

        self.semantic_lut = np.zeros((np.max([k for k in self.semantic_remap.keys()]) + 1), dtype=np.int32)
        for k, v in self.semantic_remap.items():
            self.semantic_lut[k] = v
        ##################
        # Other parameters
        ##################

        # Update number of class and data_task
        # config.num_classes  = self.num_classes
        config.num_motion_classes  = self.num_motion_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        ##################
        # Load calibration
        ##################

        # Init variables
        self.calibrations = []
        self.times = []
        self.poses = []
        self.all_inds = None
        self.class_proportions = None
        self.class_frames = []
        self.val_confs = []

        # Load everything
        self.load_calib_poses()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize frame potentials
        self.potentials = torch.from_numpy(np.random.rand(self.all_inds.shape[0]) * 0.1 + 0.1)
        self.potentials.share_memory_()

        # If true, the same amount of frames is picked per class
        self.balance_classes   = balance_classes

        # If true, we use fixed centers per frame
        self.fixed_centers = fixed_centers

        # Choose batch_num in_R and max_in_p depending on validation or training
        if self.set == 'training':
            self.batch_num = config.batch_num
            self.max_in_p = config.max_in_points
            self.in_R = config.in_radius
        else:
            self.batch_num = config.val_batch_num
            self.max_in_p = config.max_val_points
            self.in_R = config.val_radius 

        # shared epoch indices and classes (in case we want class balanced sampler)
        if set == 'training':
            N = int(np.ceil(config.epoch_steps * self.batch_num * 10))
            self.ext_keyss = None
        else:
            N = int(np.ceil(config.validation_size * self.batch_num * 10))
            self.ext_keyss = {}

        self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        self.epoch_inds = torch.from_numpy(np.zeros((N,), dtype=np.int64))
        self.epoch_labels = torch.from_numpy(np.zeros((N,), dtype=np.int32))
        self.epoch_centers= torch.from_numpy(np.zeros((N, 3), dtype=np.float32))
        self.epoch_i.share_memory_()
        self.epoch_inds.share_memory_()
        self.epoch_labels.share_memory_()
        self.epoch_centers.share_memory_()

        # self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
        # self.worker_waiting.share_memory_()
        self.worker_lock = Lock()

        self.emergency_data = None

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.frames)

    def __getitem__(self, batch_i, verbose=False):
        # balanced: we precompute the desired frame and target class;
        # robustness: we introduce while loop + random sample to avoid empty region;
        # mini_batch: we add centering & dynamic sampling radius together with padding of [100, 100, 100];
        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        fi_list = []
        p0_list = []
        s_list = []
        R_list = []
        r_inds_list = []
        r_mask_list = []
        val_labels_list = []
        batch_n = 0
        next_flag = True

        with self.worker_lock:
            # Get potential minimum
            ind = int(self.epoch_inds[self.epoch_i])
            wanted_label = int(self.epoch_labels[self.epoch_i])
            self.epoch_i += 1

        s_ind, f_ind = self.all_inds[ind]
        skip  = 1
        if f_ind < self.config.sample_range:
            skip = -1

        target_frames = [0] + list(np.sort(sample(range(self.config.sample_range-2, self.config.sample_range), self.config.n_frames)))
        f_incs = np.array(target_frames) * skip

        # pre-fetch full points and labels
        full_points_list = []
        full_labels_list = [] 
        
        for f_inc in f_incs:
            pose = self.poses[s_ind][f_ind - f_inc]
            seq_path  = join(self.path, 'sequences', self.sequences[s_ind])
            velo_file = join(seq_path, 'velodyne', self.frames[s_ind][f_ind - f_inc] + '.bin')
            frame_points = np.fromfile(velo_file, dtype=np.float32).reshape((-1, 4))

            if self.set == 'test':
                label_file = None
                sem_labels = np.zeros((frame_points.shape[0],), dtype=np.int32)
            else:
                label_file = join(seq_path, 'labels', self.frames[s_ind][f_ind - f_inc] + '.label')
                full_labels = np.fromfile(label_file, dtype=np.int32)
                sem_labels = full_labels & 0xFFFF  # semantic label in lower half
                sem_labels = self.learning_map[sem_labels]

            # get new_points in global map coordinate
            hpoints = np.hstack((frame_points[:, :3], np.ones_like(frame_points[:, :1])))
            full_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

            o_pts = None
            o_labels = None

            # In case of validation, keep the original points in memory
            if self.set in ['validation', 'test'] and f_inc == 0:
                o_pts = full_points[:, :3].astype(np.float32)
                o_labels = sem_labels.astype(np.int32)
            full_points_list.append(full_points)
            full_labels_list.append(sem_labels)

        center_ind = 0
        pose0 = self.poses[s_ind][f_ind]
        # print('')
        while len(p_list) < self.config.n_frames:
            if not next_flag: # break when the stack is full
                break
            # keep looking for target class points
            if self.balance_classes and center_ind<5:
                wanted_ind = np.random.choice(np.where(full_labels_list[0] == wanted_label)[0])
            else:
                wanted_ind = np.random.choice(full_labels_list[0].shape[0])
            if verbose: 
                print(f'------{len(p_list)} keep fetching {center_ind}th wanted label {wanted_label}, {wanted_ind}')
            wanted_center = full_points_list[0][wanted_ind, :3] # we randomly select point as new center
            center_ind   +=1
            batch_order  = center_ind
            merged_points = np.zeros((0, 3), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int32)
            merged_coords = np.zeros((0, 4), dtype=np.float32)

            # # Get center of the first frame in world coordinates
            # p_origin = np.zeros((1, 4), dtype=np.float32)
            # p_origin[0, 3] = 1

            # p_origin[0, :3] += wanted_center.flatten()
            # pose0 = self.poses[s_ind][f_ind]
            # p0 = p_origin.dot(pose0.T)[:, :3]
            # p0 = np.squeeze(p0).astype(np.float32)
            p0 = wanted_center.astype(np.float32)

            # Stack sequence
            p_seq = []
            f_seq = []
            l_seq = []
            fi_seq =[]
            p0_seq =[]
            s_seq = []
            R_seq = []
            r_inds_seq = []
            r_mask_seq = []
            val_labels_seq = []

            # Update 
            seq_n = 0
            for num_merged in range(self.config.n_frames):
                f_inc = f_incs[num_merged]
                pose = self.poses[s_ind][f_ind - f_inc]

                # get points 
                sem_labels = full_labels_list[num_merged]
                o_labels   = full_labels_list[num_merged]
                frame_points= np.copy(full_points_list[num_merged])
                o_pts      = np.copy(full_points_list[num_merged][:, :3])

                # mask points
                new_points = np.copy(frame_points)
                mask = np.sum(np.square(new_points[:, :3] - p0), axis=1) < self.in_R ** 2
                if not np.any(mask):
                    print('--------p0 is bad')
                    break

                mask_inds = np.where(mask)[0].astype(np.int32)

                # Shuffle points
                rand_order = np.random.permutation(mask_inds)
                new_points = new_points[rand_order, :3]
                sem_labels = sem_labels[rand_order]

                # We have to project in the first frame coordinates
                new_coords = new_points - pose0[:3, 3]
                new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
                new_coords = np.hstack((new_coords, frame_points[rand_order, 3:]))

                merged_points = new_points - p0.reshape(1, 3) # centering the points around p0
                merged_labels = sem_labels # TODO
                merged_coords = new_coords

                # Subsample merged frames(you may not want to subsample the points)
                # in_pts, in_fts, in_lbls = grid_subsampling(merged_points, features=merged_coords, labels=merged_labels, sampleDl=self.config.first_subsampling_dl)
                in_pts, in_fts, in_lbls = merged_points, merged_coords, merged_labels[:, np.newaxis]
                n = in_pts.shape[0]

                # Safe check
                if n < 5:
                    print('--------pts < 5')
                    break

                num_points = n # record the valid points
                # Randomly drop some points
                if n > int(self.max_in_p):
                    input_inds = np.random.choice(n, size=int(self.max_in_p), replace=False)
                    in_pts = in_pts[input_inds, :]
                    in_fts = in_fts[input_inds, :]
                    in_lbls = in_lbls[input_inds]
                    n = input_inds.shape[0]
                elif n < int(self.max_in_p):
                    target_n = int(self.max_in_p)
                    in_pts = np.concatenate([in_pts,  100 * np.ones((target_n - n, in_pts.shape[1]))], axis=0)
                    in_fts = np.concatenate([in_fts,  np.zeros((target_n - n, in_fts.shape[1]))], axis=0)
                    in_lbls= np.concatenate([in_lbls, np.zeros((target_n - n, in_lbls.shape[1]))], axis=0)
                    n = target_n                  

                # Before augmenting, compute reprojection inds (only for validation and test)
                if self.set in ['validation', 'test'] and f_inc == 0:
                    radiuses = np.sum(np.square(o_pts - p0.reshape(1, 3)), axis=1)
                    reproj_mask = radiuses < (0.99 * self.in_R) ** 2
                    if not np.any(reproj_mask):
                        print('-------reproj_mask not valid')
                        break
                    # Project predictions on the frame points
                    search_tree = KDTree(in_pts, leaf_size=50)
                    proj_inds = search_tree.query(o_pts[reproj_mask, :] - p0.reshape(1, 3), return_distance=False)
                    proj_inds = np.squeeze(proj_inds).astype(np.int32)
                else:
                    proj_inds = np.zeros((0,))
                    reproj_mask = np.zeros((0,))

                # Data augmentation
                if f_inc == 0:
                    _, scale, R = self.augmentation_transform(in_pts, return_components_only=True)

                noise = (np.random.randn(in_pts.shape[0], in_pts.shape[1]) * self.config.augment_noise).astype(np.float32)
                in_pts = np.sum(np.expand_dims(in_pts, 2) * R, axis=1) * scale + noise
                if np.random.rand() > self.config.augment_color:
                    in_fts[:, 3:] *= 0

                # Stack sequence
                p_seq += [in_pts]
                f_seq += [in_fts[:, 3:]]
                l_seq += [np.squeeze(in_lbls)]
                fi_seq += [[s_ind, f_ind]]
                p0_seq += [p0]
                s_seq += [scale]
                R_seq += [R]
                r_inds_seq += [proj_inds]
                r_mask_seq += [reproj_mask]
                val_labels_seq += [o_labels]

                # Update 
                seq_n +=n

            # Stack batch
            if len(p_seq) == self.config.n_frames:
                p_list += p_seq
                f_list += f_seq
                l_list += l_seq
                fi_list += fi_seq
                p0_list += p0_seq
                s_list += s_seq
                R_list += R_seq
                r_inds_list += r_inds_seq
                r_mask_list += r_mask_seq
                val_labels_list += val_labels_seq 
                batch_n += seq_n

        #>>>>>>>>>>>>>>>>>..
        xyz_list = [torch.from_numpy(x.astype(np.float32)) for x in p_list]
        feat_list= [torch.from_numpy(x.astype(np.float32)) for x in f_list]

        save_viz = False
        if save_viz:
            save_path = join(self.config.log_dir, 'viz')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_name = save_path + f'/{self.epoch_i.cpu().numpy()[0]}.npy'
            viz_dict = {}
            viz_dict['input_querys'] = p_list[0]
            viz_dict['input_points'] = np.concatenate(p_list, axis=0)
            # viz_dict['input_neighbors'] = input_neighbors
            # viz_dict['input_pools']  = input_pools
            # viz_dict['input_upsamples'] = input_upsamples
            # viz_dict['input_stack_lengths'] = input_stack_lengths
            viz_dict['label'] = l_list[0]
            viz_dict['label_points'] = np.concatenate(l_list, axis=0)

            for key, value in viz_dict.items():
                print(key, len(value))

            print('saving to ', save_name)
            np.save(save_name, arr=viz_dict)

        if self.config.arch_decoder == 'meteornet':
            # breakpoint()
            xyzs = torch.cat(xyz_list, dim=0)
            feats= torch.cat(feat_list, dim=0)
            times= torch.ones(self.nframe, self.max_in_p).float() * torch.arange(0, self.nframe).unsqueeze(-1)
            times= times.view(1, -1).contiguous()

            semantic_mask = np.ones((xyzs.shape[0]), dtype=np.float32)
            ignore_ind    = np.where(np.concatenate(l_list, axis=0).astype(np.int32)==0)[0]
            semantic_mask[ignore_ind] = 0
            semantic_mask = torch.from_numpy(semantic_mask.copy().astype(np.float32)).unsqueeze(0)
            if self.config.num_classes == 20:
                unproj_labels= torch.from_numpy(self.semantic_lut[np.concatenate(l_list, axis=0).astype(np.int32).tolist()])
            else:
                unproj_labels= torch.from_numpy(np.concatenate(l_list, axis=0).astype(np.int32))
            unproj_motion_state = torch.from_numpy(self.motion_lut[np.concatenate(l_list, axis=0).astype(np.int32).tolist()])

            motion_mask   = np.ones((xyzs.shape[0]), dtype=np.float32)
            ignore_ind    = np.where(unproj_motion_state.numpy().astype(np.int32)==0)[0]
            motion_mask[ignore_ind] = 0
            motion_mask = torch.from_numpy(motion_mask.copy().astype(np.float32)).unsqueeze(0)

            return xyz_list[0].permute(1, 0), xyzs.permute(1, 0), times, feats.permute(1, 0), semantic_mask, unproj_labels, motion_mask, unproj_motion_state

        else:
            semantic_mask = np.ones((p_list[0].shape[0]), dtype=np.float32)
            ignore_ind    = np.where(l_list[0]==0)[0]
            semantic_mask[ignore_ind] = 0
            semantic_mask = torch.from_numpy(semantic_mask.copy().astype(np.float32)).unsqueeze(0)

            motion_mask = np.ones((p_list[0].shape[0]), dtype=np.float32)
            ignore_ind    = np.where(self.motion_lut[l_list[0].astype(np.int32).tolist()]==0)[0]
            motion_mask[ignore_ind] = 0
            motion_mask = torch.from_numpy(motion_mask.copy().astype(np.float32)).unsqueeze(0)
            if self.config.num_classes == 20: 
                unproj_labels= torch.from_numpy(self.semantic_lut[l_list[0].astype(np.int32).tolist()])
            else:
                unproj_labels= torch.from_numpy(l_list[0].astype(np.int32))
            unproj_motion_state = torch.from_numpy(self.motion_lut[l_list[0].astype(np.int32).tolist()])
            index_seq_frame = np.array(fi_list)
            
            xyzs = torch.stack(xyz_list, 0)
            feats= torch.stack(feat_list,0)
            times= torch.ones(self.nframe, self.max_in_p).float() * torch.arange(0, self.nframe).unsqueeze(-1)
            times= times.unsqueeze(1)

            if verbose: 
                print('')
                print(f'xyzs: {xyzs.size()}, times: {times.size()}, feats: {feats.size()}, semantic_mask: {semantic_mask.size()}')

            return xyz_list[0].permute(1, 0), xyzs.permute(0, 2, 1), times, feats.permute(0, 2, 1), semantic_mask, unproj_labels, motion_mask, unproj_motion_state

    def __getitem_external__(self, batch_i, ext_keys, verbose=False):
        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        fi_list = []
        p0_list = []
        s_list = []
        R_list = []
        r_inds_list = []
        r_mask_list = []
        val_labels_list = []
        batch_n = 0
        next_flag = True

        # pre-set squence and frame number
        s_ind, f_ind = ext_keys['s_ind'], ext_keys['f_ind']
        skip  = 1
        if f_ind < self.config.sample_range:
            skip = -1

        target_frames = [0] + list(np.sort(sample(range(self.config.sample_range-2, self.config.sample_range), self.config.n_frames)))
        f_incs = np.array(target_frames) * skip

        # pre-fetch full points and labels
        full_points_list = []
        full_labels_list = [] # 
        
        for f_inc in f_incs:
            pose = self.poses[s_ind][f_ind - f_inc]
            seq_path  = join(self.path, 'sequences', self.sequences[s_ind])
            velo_file = join(seq_path, 'velodyne', self.frames[s_ind][f_ind - f_inc] + '.bin')
            frame_points = np.fromfile(velo_file, dtype=np.float32).reshape((-1, 4))

            if self.set == 'test':
                label_file = None
                sem_labels = np.zeros((frame_points.shape[0],), dtype=np.int32)
            else:
                label_file = join(seq_path, 'labels', self.frames[s_ind][f_ind - f_inc] + '.label')
                full_labels = np.fromfile(label_file, dtype=np.int32)
                sem_labels = full_labels & 0xFFFF  # semantic label in lower half
                sem_labels = self.learning_map[sem_labels]

            # get new_points in global map coordinate
            hpoints = np.hstack((frame_points[:, :3], np.ones_like(frame_points[:, :1])))
            full_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

            # o_pts = None
            # o_labels = None

            # # In case of validation, keep the original points in memory
            # if self.set in ['validation', 'test'] and f_inc == 0:
            #     o_pts = full_points[:, :3].astype(np.float32)
            #     o_labels = sem_labels.astype(np.int32)
            full_points_list.append(full_points)
            full_labels_list.append(sem_labels)

        while True:
            if not next_flag: # break when the stack is full
                break
            if ext_keys['m'] > ext_keys['centers'].shape[0] - 1: # break when this frame is over
                if len(p_list) == 0: 
                    return None
                else: 
                    break
            # new sample center
            wanted_center= ext_keys['centers'][ext_keys['m'], :]
            center_ind   = ext_keys['m']
            batch_order  = center_ind

            if verbose: 
                print(f'looping over {center_ind}th center: {wanted_center}')
            ext_keys['m']+=1 

            merged_points = np.zeros((0, 3), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int32)
            merged_coords = np.zeros((0, 4), dtype=np.float32)

            # Get center of the first frame in world coordinates
            p_origin = np.zeros((1, 4), dtype=np.float32)
            p_origin[0, 3] = 1

            p_origin[0, :3] += wanted_center.flatten()
            pose0 = self.poses[s_ind][f_ind]
            p0 = p_origin.dot(pose0.T)[:, :3]
            p0 = np.squeeze(p0).astype(np.float32)

            # Stack sequence
            p_seq = []
            f_seq = []
            l_seq = []
            fi_seq = []
            p0_seq = []
            s_seq = []
            R_seq = []
            r_inds_seq = []
            r_mask_seq = []
            val_labels_seq = []

            # Update 
            seq_n = 0
            for num_merged in range(self.config.n_frames):
                f_inc = target_frames[num_merged] * skip
                pose = self.poses[s_ind][f_ind - f_inc]

                # get points 
                sem_labels = full_labels_list[num_merged]
                o_labels   = full_labels_list[num_merged]
                frame_points= np.copy(full_points_list[num_merged])
                o_pts      = np.copy(full_points_list[num_merged][:, :3])

                # mask points too far
                new_points = np.copy(frame_points)
                mask = np.sum(np.square(new_points[:, :3] - p0), axis=1) < self.in_R ** 2
                if not np.any(mask):
                    # print(f'jumping {batch_order}') # jump all frames
                    break
                mask_inds = np.where(mask)[0].astype(np.int32)

                # Shuffle points
                rand_order = np.random.permutation(mask_inds)
                new_points = new_points[rand_order, :3]
                sem_labels = sem_labels[rand_order]

                # We have to project in the first frame coordinates
                new_coords = new_points - pose0[:3, 3]
                new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
                new_coords = np.hstack((new_coords, frame_points[rand_order, 3:]))

                merged_points = new_points - p0.reshape(1, 3) # centering the points
                merged_labels = sem_labels # TODO
                merged_coords = new_coords

                # Subsample merged frames
                in_pts, in_fts, in_lbls = grid_subsampling(merged_points, features=merged_coords, labels=merged_labels, sampleDl=self.config.first_subsampling_dl)

                n = in_pts.shape[0]

                # Safe check
                if n < 5:
                    # print(f'jumping {batch_order}')
                    break

                if len(p_list)>self.config.n_frames:
                    if batch_n + seq_n + (self.config.n_frames- num_merged) * min(self.config.max_frame_points, n) > self.config.max_gpu_points:
                        next_flag = False
                        break

                # Randomly drop some points
                if n > int(self.config.max_frame_points):
                    input_inds = np.random.choice(n, size=int(self.config.max_frame_points), replace=False)
                    in_pts = in_pts[input_inds, :]
                    in_fts = in_fts[input_inds, :]
                    in_lbls = in_lbls[input_inds]
                    n = input_inds.shape[0]

                # Before augmenting, compute reprojection inds (only for validation and test)
                if self.set in ['validation', 'test'] and f_inc == 0:
                    # get val_points that are in range
                    radiuses = np.sum(np.square(o_pts - p0.reshape(1, 3)), axis=1)
                    reproj_mask = radiuses < (0.99 * self.in_R) ** 2
                    if not np.any(reproj_mask):
                        # print(f'jumping {batch_order}')
                        break

                    # Project predictions on the frame points
                    search_tree = KDTree(in_pts, leaf_size=50)
                    proj_inds = search_tree.query(o_pts[reproj_mask, :] - p0.reshape(1, 3), return_distance=False)
                    proj_inds = np.squeeze(proj_inds).astype(np.int32)
                else:
                    proj_inds = np.zeros((0,))
                    reproj_mask = np.zeros((0,))

                # Data augmentation
                if f_inc == 0:
                    _, scale, R = self.augmentation_transform(in_pts, return_components_only=True)
                noise = (np.random.randn(in_pts.shape[0], in_pts.shape[1]) * self.config.augment_noise).astype(np.float32)
                in_pts = np.sum(np.expand_dims(in_pts, 2) * R, axis=1) * scale + noise
                # Color augmentation
                if np.random.rand() > self.config.augment_color:
                    in_fts[:, 3:] *= 0

                # Stack sequence
                p_seq += [in_pts]
                f_seq += [in_fts]
                l_seq += [np.squeeze(in_lbls)]
                fi_seq += [[s_ind, f_ind]]
                p0_seq += [p0]
                s_seq += [scale]
                R_seq += [R]
                r_inds_seq += [proj_inds]
                r_mask_seq += [reproj_mask]
                val_labels_seq += [o_labels]

                # Update 
                seq_n +=n

            # Stack batch
            if len(p_seq) == self.config.n_frames:
                p_list += p_seq
                f_list += f_seq
                l_list += l_seq
                fi_list += fi_seq
                p0_list += p0_seq
                s_list += s_seq
                R_list += R_seq
                r_inds_list += r_inds_seq
                r_mask_list += r_mask_seq
                val_labels_list += val_labels_seq 
                batch_n += seq_n

            if batch_n > self.config.max_safe_points:
                print(f'we got {batch_n} input points')
                break

        ######### stack batch 
        xyz1 = sampled_pts_xyz[0]
        xyz2 = torch.stack(sampled_pts_xyz, 0)
        feat1 = sampled_pts_feat[0]
        feat2 = torch.stack(sampled_pts_feat, 0)
        ###################
        # Concatenate batch
        ###################
        # stacked_points = np.concatenate(p_list, axis=0).astype(np.float32)
        # features = np.concatenate(f_list, axis=0)
        # labels   = np.concatenate(l_list, axis=0)
        # frame_inds = np.array(fi_list, dtype=np.int32)
        # frame_centers = np.stack(p0_list, axis=0)
        # stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        # scales = np.array(s_list, dtype=np.float32)
        # rots = np.stack(R_list, axis=0)

        # Input features (Use reflectance, input height or all coordinates)
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 2:
            # Use original height coordinate
            stacked_features = np.hstack((stacked_features, features[:, 2:3]))
        elif self.config.in_features_dim == 3:
            # Use height + reflectance
            stacked_features = np.hstack((stacked_features, features[:, 2:]))
        elif self.config.in_features_dim == 4:
            # Use all coordinates
            stacked_features = np.hstack((stacked_features, features[:, :3])) # TODO
        elif self.config.in_features_dim == 5:
            # Use all coordinates + reflectance
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        # return [self.config.num_layers] + input_list
        return xyz1, xyz2, feat1, feat2, semantic_mask, unproj_labels, unproj_motion_state

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in self.sequences:

            seq_folder = join(self.path, 'sequences', seq)

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

        ###################################
        # Prepare the indices of all frames
        ###################################

        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.frames)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.frames])
        self.all_inds = np.vstack((seq_inds, frame_inds)).T

        ################################################
        # For each class list the frames containing them
        ################################################

        if self.set in ['training', 'validation']:

            class_frames_bool = np.zeros((0, self.num_classes), dtype=np.bool)
            self.class_proportions = np.zeros((self.num_classes,), dtype=np.int32)

            for s_ind, (seq, seq_frames) in enumerate(zip(self.sequences, self.frames)):

                frame_mode = 'single'
                if self.config.n_frames > 1:
                    frame_mode = 'multi'
                seq_stat_file = join(self.path, 'sequences', seq, '{}_stats_{:s}.pkl'.format(self.num_classes, frame_mode))

                # Check if inputs have already been computed
                if exists(seq_stat_file):
                    # Read pkl
                    print('reading ', seq_stat_file)
                    with open(seq_stat_file, 'rb') as f:
                        seq_class_frames, seq_proportions = pickle.load(f)
                else:

                    # Initiate dict
                    print('Preparing seq {:s} class frames. (Long but one time only)'.format(seq))

                    # Class frames as a boolean mask
                    seq_class_frames = np.zeros((len(seq_frames), self.num_classes), dtype=np.bool)

                    # Proportion of each class
                    seq_proportions = np.zeros((self.num_classes,), dtype=np.int32)

                    # Sequence path
                    seq_path = join(self.path, 'sequences', seq)

                    # Read all frames
                    for f_ind, frame_name in enumerate(seq_frames):

                        # Path of points and labels
                        label_file = join(seq_path, 'labels', frame_name + '.label')

                        # Read labels
                        frame_labels = np.fromfile(label_file, dtype=np.int32)
                        sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                        sem_labels = self.learning_map[sem_labels]

                        # Get present labels and there frequency
                        unique, counts = np.unique(sem_labels, return_counts=True)

                        # Add this frame to the frame lists of all class present
                        frame_labels = np.array([self.label_to_idx[l] for l in unique], dtype=np.int32)
                        seq_class_frames[f_ind, frame_labels] = True

                        # Add proportions
                        seq_proportions[frame_labels] += counts

                    # Save pickle
                    with open(seq_stat_file, 'wb') as f:
                        pickle.dump([seq_class_frames, seq_proportions], f)

                class_frames_bool = np.vstack((class_frames_bool, seq_class_frames))
                self.class_proportions += seq_proportions

            # Transform boolean indexing to int indices.
            self.class_frames = []
            for i, c in enumerate(self.label_values):
                if c in self.ignored_labels:
                    self.class_frames.append(torch.zeros((0,), dtype=torch.int64))
                else:
                    integer_inds = np.where(class_frames_bool[:, i])[0]
                    self.class_frames.append(torch.from_numpy(integer_inds.astype(np.int64)))

        # Add variables for validation
        if self.set == 'validation':
            self.val_points = []
            self.val_labels = []
            self.val_confs = []

            for s_ind, seq_frames in enumerate(self.frames):
                if 'motion' in self.dataset_task:
                    self.val_confs.append(np.zeros((len(seq_frames), self.num_motion_classes, self.num_motion_classes))) # per sequence
                else:
                    self.val_confs.append(np.zeros((len(seq_frames), self.num_classes, self.num_classes)))

        return


    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses
    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]
# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/

#sssam
class SemanticKittiSampler(Sampler):
    """Sampler for SemanticKitti"""
    def __init__(self, dataset: SemanticKittiDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        # Initiate current epoch ind
        self.dataset.epoch_i *= 0
        self.dataset.epoch_inds *= 0
        self.dataset.epoch_labels *= 0
        self.dataset.epoch_centers *=0

        # Number of sphere centers taken per class in each cloud
        num_centers = self.dataset.epoch_inds.shape[0]
        print('num_centers are : ', num_centers)

        if self.dataset.balance_classes:
            # Generate a list of indices balancing classes and respecting potentials
            gen_indices = []
            gen_classes = []
            bg_classes  = [0]
            for i, c in enumerate(self.dataset.label_values): # TODO
                if c not in bg_classes: # focus on objects
                    class_potentials = self.dataset.potentials[self.dataset.class_frames[i]]
                    print('potentials for class: ', c, class_potentials.shape[0])
                    # Get the indices to generate thanks to potentials
                    used_classes = self.dataset.num_classes - len(self.dataset.ignored_labels)
                    # class_n = num_centers // used_classes + 1
                    class_n = num_centers // (len(self.dataset.label_values)-1) + 1 # 
                    if class_n < class_potentials.shape[0]:
                        _, class_indices = torch.topk(class_potentials, class_n, largest=False)
                    else:
                        class_indices = torch.zeros((0,), dtype=torch.int32)
                        while class_indices.shape[0] < class_n:
                            new_class_inds = torch.randperm(class_potentials.shape[0])
                            class_indices = torch.cat((class_indices.type(torch.LongTensor), new_class_inds), dim=0)
                        class_indices = class_indices[:class_n]
                    class_indices = self.dataset.class_frames[i][class_indices]
                    gen_indices.append(class_indices)
                    gen_classes.append(class_indices * 0 + c)

                    # Update potentials
                    update_inds = torch.unique(class_indices)
                    self.dataset.potentials[update_inds] = torch.ceil(self.dataset.potentials[update_inds])
                    self.dataset.potentials[update_inds] += torch.from_numpy(np.random.rand(update_inds.shape[0]) * 0.1 + 0.1) # increase the potentials

            # Stack the chosen indices of all classes
            gen_indices = torch.cat(gen_indices, dim=0)
            gen_classes = torch.cat(gen_classes, dim=0)

            # Shuffle generated indices
            rand_order = torch.randperm(gen_indices.shape[0])[:num_centers]
            gen_indices = gen_indices[rand_order]
            gen_classes = gen_classes[rand_order]

            # Update epoch inds
            self.dataset.epoch_inds[:len(gen_indices)]   += gen_indices
            self.dataset.epoch_labels[:len(gen_classes)] += gen_classes.type(torch.int32)
        else:
            # Get the list of indices to generate thanks to potentials
            if num_centers < self.dataset.potentials.shape[0]:
                _, gen_indices = torch.topk(self.dataset.potentials, num_centers, largest=False, sorted=True)
            else:
                gen_indices = torch.randperm(self.dataset.potentials.shape[0])

            # Update potentials (Change the order for the next epoch)
            self.dataset.potentials[gen_indices] = torch.ceil(self.dataset.potentials[gen_indices])
            self.dataset.potentials[gen_indices] += torch.from_numpy(np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1)

            # Update epoch inds
            self.dataset.epoch_inds[:len(gen_indices)]  += gen_indices

        self.dataset.epoch_centers *= 0

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    # sscal
    def calib_max_in(self, config, dataloader, untouched_ratio=0.8, verbose=True, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration of max_in_points value (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load max_in_limit dictionary
        max_in_lim_file = join(self.dataset.path, 'max_in_limits.pkl')
        if exists(max_in_lim_file):
            with open(max_in_lim_file, 'rb') as file:
                max_in_lim_dict = pickle.load(file)
        else:
            max_in_lim_dict = {}

        print('checking max_in_lim_dict')
        print(max_in_lim_dict)
        #breakpoint()
        # Check if the max_in limit associated with current parameters exists
        if self.dataset.balance_classes:
            sampler_method = 'balanced'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}'.format(sampler_method,
                                          self.dataset.in_R,
                                          self.dataset.config.first_subsampling_dl)
        if not redo and key in max_in_lim_dict:
            self.dataset.max_in_p = max_in_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check max_in limit dictionary')
            if key in max_in_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(max_in_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:
            ########################
            # Batch calib parameters
            ########################

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            all_lengths = []
            N = 1000

            #####################
            # Perform calibration
            #####################
            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):
                    if batch.points is None:
                        continue 
                    # Control max_in_points value
                    all_lengths += batch.lengths[0].tolist()

                    # Convergence
                    if len(all_lengths) > N:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if t - last_display > 1.0:
                        last_display = t
                        message = 'Collecting {:d} in_points: {:5.1f}%'
                        print(message.format(N,
                                             100 * len(all_lengths) / N))

                if breaking:
                    break

            self.dataset.max_in_p = int(np.percentile(all_lengths, 100*untouched_ratio))

            if verbose:
                # Create histogram
                a = 1

            # Save max_in_limit dictionary
            print('New max_in_p = ', self.dataset.max_in_p)
            max_in_lim_dict[key] = self.dataset.max_in_p
            with open(max_in_lim_file, 'wb') as file:
                pickle.dump(max_in_lim_dict, file)

        # Update value in config
        if self.dataset.set == 'training':
            config.max_in_points = self.dataset.max_in_p
        else:
            config.max_val_points = self.dataset.max_in_p

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return

class Parser():
    def __init__(self, cfg, DATA):
        super(Parser, self).__init__()

        self.labels = DATA["labels"]
        self.learning_ignore = DATA["learning_ignore"]
        self.learning_map = DATA["learning_map"]
        self.learning_map_inv = DATA["learning_map_inv"]
        self.color_map = DATA["color_map"]

        # Initialize datasets
        self.train_dataset    = SemanticKittiDataset(cfg, set='training', balance_classes=True)
        self.valid_dataset    = SemanticKittiDataset(cfg, set='validation', balance_classes=False)
        self.test_dataset     = SemanticKittiDataset(cfg, set='test', balance_classes=False)

        # Initialize samplers
        training_sampler = SemanticKittiSampler(self.train_dataset)
        valid_sampler = SemanticKittiSampler(self.valid_dataset)
        test_sampler = SemanticKittiSampler(self.test_dataset)

        # Initialize the dataloader
        self.trainloader = DataLoader(self.train_dataset,
                                     batch_size=cfg.TRAIN.batch_size,
                                     sampler=training_sampler,
                                     num_workers=cfg.input_threads,
                                     pin_memory=True)

        if cfg.debug: 
            datapoint = self.valid_dataset.__getitem__(200)
            for single_item in datapoint: 
                try: 
                    print(single_item.shape)
                except: 
                    pass

        self.validloader = DataLoader(self.valid_dataset,
                                 batch_size=cfg.TRAIN.batch_size,
                                 sampler=valid_sampler,
                                 num_workers=cfg.input_threads,
                                 pin_memory=True)

        self.testloader = DataLoader(self.test_dataset,
                                 batch_size=cfg.TRAIN.batch_size,
                                 sampler=test_sampler,
                                 num_workers=cfg.input_threads,
                                 pin_memory=True)

        # Calibrate max_in_point value, per frame
        training_sampler.calib_max_in(cfg, self.trainloader, verbose=True)
        valid_sampler.calib_max_in(cfg, self.validloader, verbose=True)

        # # Calibrate samplers, max_gpu_points, and 
        # training_sampler.calibration(training_loader, verbose=True)
        # training_sampler.dataset.neighborhood_limits = [32, 32, 32, 32, 32]
        # valid_sampler.calibration(valid_loader, verbose=True)
        # valid_sampler.dataset.neighborhood_limits = [32, 32, 32, 32, 32] 
        # test_sampler.calibration(test_loader, verbose=True)

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        # put label in original values
        return SemanticKittiDataset.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        # put label in xentropy values
        return SemanticKittiDataset.map(label, self.learning_map)

    def to_motion(self, label):
        # put label in xentropy values
        return SemanticKittiDataset.map(label, self.valid_dataset.motion_map)

    def to_color(self, label):
        # put label in original values
        label = SemanticKittiDataset.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKittiDataset.map(label, self.color_map)

# class SemanticKittiCustomBatch:
#     """Custom batch definition with memory pinning for SemanticKitti"""

#     def __init__(self, input_list):

#         # Get rid of batch dimension
#         if input_list[0] is None: 
#             self.points = None
#             return 
#         input_list = input_list[0]

#         # Number of layers
#         L = int(input_list[0])

#         # Extract input tensors from the list of numpy array
#         ind = 1
#         self.querys = [torch.from_numpy(nparray).float() for nparray in input_list[ind:ind+L]]
#         ind += L
#         self.points = [torch.from_numpy(nparray).float() for nparray in input_list[ind:ind+L]]
#         ind += L
#         self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
#         ind += L
#         self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
#         ind += L
#         self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
#         ind += L
#         self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
#         ind += L
#         self.features = torch.from_numpy(input_list[ind]).float()
#         ind += 1
#         self.labels = torch.from_numpy(input_list[ind])
#         ind += 1
#         self.scales = torch.from_numpy(input_list[ind])
#         ind += 1
#         self.rots = torch.from_numpy(input_list[ind])
#         ind += 1
#         self.frame_inds = torch.from_numpy(input_list[ind])
#         ind += 1
#         self.frame_centers = torch.from_numpy(input_list[ind])
#         ind += 1
#         self.reproj_inds = input_list[ind]
#         ind += 1
#         self.reproj_masks= input_list[ind]
#         ind += 1
#         self.val_labels  = input_list[ind]

#         return

#     def pin_memory(self):
#         """
#         Manual pinning of the memory
#         """
#         if self.points is None:
#             return self
#         self.querys = [in_tensor.pin_memory() for in_tensor in self.querys]
#         self.points = [in_tensor.pin_memory() for in_tensor in self.points]
#         self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
#         self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
#         self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
#         self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
#         self.features = self.features.pin_memory()
#         self.labels = self.labels.pin_memory()
#         self.scales = self.scales.pin_memory()
#         self.rots = self.rots.pin_memory()
#         self.frame_inds = self.frame_inds.pin_memory()
#         self.frame_centers = self.frame_centers.pin_memory()

#         return self

#     def to(self, device):
#         if self.points is None:
#             return self
#         self.querys = [in_tensor.to(device) for in_tensor in self.querys]
#         self.points = [in_tensor.to(device) for in_tensor in self.points]
#         self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
#         self.pools = [in_tensor.to(device) for in_tensor in self.pools]
#         self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
#         self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
#         self.features = self.features.to(device)
#         self.labels = self.labels.to(device)
#         self.scales = self.scales.to(device)
#         self.rots = self.rots.to(device)
#         self.frame_inds = self.frame_inds.to(device)
#         self.frame_centers = self.frame_centers.to(device)

#         return self

#     def unstack_points(self, layer=None):
#         """Unstack the points"""
#         return self.unstack_elements('points', layer)

#     def unstack_neighbors(self, layer=None):
#         """Unstack the neighbors indices"""
#         return self.unstack_elements('neighbors', layer)

#     def unstack_pools(self, layer=None):
#         """Unstack the pooling indices"""
#         return self.unstack_elements('pools', layer)

#     def unstack_elements(self, element_name, layer=None, to_numpy=True):
#         """
#         Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
#         layers
#         """
#         if self.points is None:
#             return None
#         if element_name == 'points':
#             elements = self.points
#         elif element_name == 'neighbors':
#             elements = self.neighbors
#         elif element_name == 'pools':
#             elements = self.pools[:-1]
#         else:
#             raise ValueError('Unknown element name: {:s}'.format(element_name))

#         all_p_list = []
#         for layer_i, layer_elems in enumerate(elements):

#             if layer is None or layer == layer_i:

#                 i0 = 0
#                 p_list = []
#                 if element_name == 'pools':
#                     lengths = self.lengths[layer_i+1]
#                 else:
#                     lengths = self.lengths[layer_i]

#                 for b_i, length in enumerate(lengths):

#                     elem = layer_elems[i0:i0 + length]
#                     if element_name == 'neighbors':
#                         elem[elem >= self.points[layer_i].shape[0]] = -1
#                         elem[elem >= 0] -= i0
#                     elif element_name == 'pools':
#                         elem[elem >= self.points[layer_i].shape[0]] = -1
#                         elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
#                     i0 += length

#                     if to_numpy:
#                         p_list.append(elem.numpy())
#                     else:
#                         p_list.append(elem)

#                 if layer == layer_i:
#                     return p_list

#                 all_p_list.append(p_list)

        # return all_p_list


# def SemanticKittiCollate(batch_data):
#     return SemanticKittiCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


# def debug_timing(dataset, loader):
#     """Timing of generator function"""

#     t = [time.time()]
#     last_display = time.time()
#     mean_dt = np.zeros(2)
#     estim_b = dataset.batch_num
#     estim_N = 0

#     for epoch in range(10):

#         for batch_i, batch in enumerate(loader):
#             # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

#             # New time
#             t = t[-1:]
#             t += [time.time()]

#             # Update estim_b (low pass filter)
#             estim_b += (len(batch.frame_inds) - estim_b) / 100
#             estim_N += (batch.features.shape[0] - estim_N) / 10

#             # Pause simulating computations
#             time.sleep(0.05)
#             t += [time.time()]

#             # Average timing
#             mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

#             # Console display (only one per second)
#             if (t[-1] - last_display) > -1.0:
#                 last_display = t[-1]
#                 message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
#                 print(message.format(batch_i,
#                                      1000 * mean_dt[0],
#                                      1000 * mean_dt[1],
#                                      estim_b,
#                                      estim_N))

#         print('************* Epoch ended *************')

#     _, counts = np.unique(dataset.input_labels, return_counts=True)
#     print(counts)


# def debug_class_w(dataset, loader):
#     """Timing of generator function"""

#     i = 0

#     counts = np.zeros((dataset.num_classes,), dtype=np.int64)

#     s = '{:^6}|'.format('step')
#     for c in dataset.label_names:
#         s += '{:^6}'.format(c[:4])
#     print(s)
#     print(6*'-' + '|' + 6*dataset.num_classes*'-')

#     for epoch in range(10):
#         for batch_i, batch in enumerate(loader):
#             # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

#             # count labels
#             new_counts = np.bincount(batch.labels)

#             counts[:new_counts.shape[0]] += new_counts.astype(np.int64)

#             # Update proportions
#             proportions = 1000 * counts / np.sum(counts)

#             s = '{:^6d}|'.format(i)
#             for pp in proportions:
#                 s += '{:^6.1f}'.format(pp)
#             print(s)
#             i += 1

if __name__ == '__main__':
    # test data 
    print('Con!')