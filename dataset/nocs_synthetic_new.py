import numpy as np
import hydra
import random
import os
import glob
import scipy.io as sio
import torch
import pickle
import cv2
import torch.utils.data as data
from os.path import join as pjoin
import matplotlib.pyplot as plt
import __init__
try:
    import vgtk.so3conv.functional as L
    import vgtk.pc as pctk
except:
    pass
from dataset.modelnet40new_render import backproject

def bp():
    import pdb;pdb.set_trace()

def rotation_distance_np(r0, r1):
    '''
    tip: r1 is usally the anchors
    '''
    if r0.ndim == 3:
        bidx = np.zeros(r0.shape[0]).astype(np.int32)
        traces = np.zeros([r0.shape[0], r1.shape[0]]).astype(np.int32)
        for bi in range(r0.shape[0]):
            diff_r = np.matmul(r1, r0[bi].T)
            traces[bi] = np.einsum('bii->b', diff_r)
            bidx[bi] = np.argmax(traces[bi])
        return traces, bidx
    else:
        # diff_r = np.matmul(r0, r1.T)
        # return np.einsum('ii', diff_r)

        diff_r = np.matmul(np.transpose(r1,(0,2,1)), r0)
        traces = np.einsum('bii->b', diff_r)

        return traces, np.argmax(traces), diff_r

def get_index(src_length, tgt_length):
    idx = np.arange(0, src_length)
    if src_length < tgt_length:
        idx = np.pad(idx, (0, tgt_length - src_length), 'wrap')
    idx = np.random.permutation(idx)[:tgt_length]
    return idx


def backproject_nocs(depth, intrinsics=np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]),
                     mask=None, scale=0.001):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    height = image_shape[0]

    non_zero_mask = (depth > 0)
    if mask is not None:
        final_instance_mask = np.logical_and(mask, non_zero_mask)
    else:
        final_instance_mask = non_zero_mask

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], height - idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]].astype(np.float32)

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 2] = -pts[:, 2]   # x, y is divided by |z| during projection --> here depth > 0 = |z| = -z

    return pts * scale, idxs


def get_nocs_data(data_path, instance, track_name, prefix, intrinsics, real=False):
    file_path = pjoin(data_path, track_name)
    try:
        depth = cv2.imread(pjoin(file_path, f'{prefix}_depth.png'), -1)
        mask = cv2.imread(pjoin(file_path, f'{prefix}_mask.png'))[:, :, 2]
        with open(pjoin(file_path, f'{prefix}_meta.txt'), 'r') as f:
            meta_lines = f.readlines()
        with open(pjoin(file_path, f'{prefix}_pose.pkl'), 'rb') as f:
            pose_dict = pickle.load(f)
    except:
        return None,  None
    if not real:
        depth, mask = depth[:, ::-1], mask[:, ::-1]
        inst_num = -1
        for meta_line in meta_lines:
            inst_num = int(meta_line.split()[0])
            inst_id = meta_line.split()[-1]
            if inst_id == instance:
                break
        if inst_num not in pose_dict:
            return None, None
        pose = pose_dict[inst_num]
        pts, _ = backproject_nocs(depth, intrinsics=intrinsics, mask=(mask == inst_num))
        return pts, pose


class NOCSDatasetNew(data.Dataset):
    def __init__(self, cfg, root, mode=None):
        super(NOCSDatasetNew, self).__init__()
        self.cfg = cfg
        self.mode = cfg.mode if mode is None else mode
        self.dataset_path = root
        self.category = cfg.target_category
        self.category_num = {'bottle': '1', 'bowl': '2', 'camera': '3',
                             'can': '4', 'laptop': '5', 'mug': '6'}[self.category]
        self.data_list_path = pjoin(self.dataset_path, 'meta', 'data_list', self.mode, self.category_num)
        self.points_path = pjoin(self.dataset_path, 'meta', 'model_pts')
        self.render_path = pjoin(self.dataset_path, 'nocs_full', self.mode)
        self.intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]]) if 'real' in self.mode else np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
        if cfg.use_fps_points:
            self.num_points = 4 * cfg.num_points
            print(f'---using {self.num_points} points as input')
        else:
            self.num_points = cfg.num_points

        scale_dict = {'bottle': 0.5, 'bowl': 0.25, 'camera': 0.27,
                      'can': 0.2, 'laptop': 0.5, 'mug': 0.21}
        self.scale_factor = scale_dict[cfg.target_category.split('_')[0]]
        self.instance_points, self.all_data = self.collect_data()
        try:
            self.anchors = L.get_anchors()
        except:
            self.anchors = np.random.rand(60, 3, 3)
        print(f"[Dataloader] : {self.mode} dataset size:", len(self.all_data))

    def collect_data(self):
        data_list = []
        instances = [f.split('.')[-2] for f in os.listdir(self.data_list_path) if f.endswith('.txt')]
        instance_points = {}
        for instance in instances:
            with open(pjoin(self.data_list_path, f'{instance}.txt')) as f:
                lines = f.readlines()
            items = [f'{instance}/{line.strip()}' for line in lines if len(line.strip())]
            data_list += items
            points_path = pjoin(self.points_path, f'{instance}.npy')
            instance_points[instance] = np.load(points_path)
        return instance_points, data_list

    def get_complete_cloud(self, instance):
        pts = self.instance_points[instance]
        idx = get_index(len(pts), self.num_points)
        return pts[idx]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        filename = self.all_data[index]
        instance, idx0, idx1 = filename.split('/')
        model_points = self.get_complete_cloud(instance)
        model_points = model_points + 0.5

        cloud, gt_pose = get_nocs_data(self.render_path, instance, idx0, idx1,
                                       self.intrinsics, 'real' in self.mode)
        if cloud is None:
            return self.__getitem__(np.random.randint(0, self.__len__()))
        target_s = gt_pose['scale']
        target_r = gt_pose['rotation']
        target_t = gt_pose['translation']
        canon_cloud = np.dot(cloud - target_t, target_r) / target_s + 0.5
        scale_norm = target_s if self.cfg.normalize_scale else self.scale_factor
        cloud = cloud / scale_norm
        target_t = target_t / scale_norm

        _, R_label, R0 = rotation_distance_np(target_r, self.anchors)

        R_gt = torch.from_numpy(target_r.astype(np.float32))  # predict r
        T = torch.from_numpy(target_t.reshape(1, 3).astype(np.float32))

        if self.cfg.eval and self.cfg.pre_compute_delta:
            cloud = model_points - 0.5
            R_gt  = torch.from_numpy(np.eye(3).astype(np.float32)) #
            T     = torch.from_numpy(np.zeros((1, 3)).astype(np.float32)) #

        data_dict = {
            'xyz': torch.from_numpy(cloud.astype(np.float32)),  # point cloud in camera space
            'points': torch.from_numpy(canon_cloud.astype(np.float32)),  # canonicalized xyz, in [0, 1]^3
            'full': torch.from_numpy(model_points.astype(np.float32)), # complete point cloud, in [0, 1]^3
            'label': torch.from_numpy(np.array([1]).astype(np.float32)),  # useless
            'R_gt': R_gt,
            'R_label': R_label,
            'R': torch.from_numpy(R0.astype(np.float32)),
            'T': T,
            'fn': self.all_data[index],
            'id': instance,
            'idx': index,
            'class': self.category
        }

        return data_dict


@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    dataset = NOCSDatasetNew(cfg, 'test')
    print('length', len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]


if __name__ == '__main__':
    main()
