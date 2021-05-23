#!/usr/bin/env python3
import os
import numpy as np
import cv2
import random
import torch
import yaml
from os.path import join as pjoin
# from common.yj_pose import rot_diff_degree, rotate_about_axis

def rot_diff_rad(rot1, rot2, chosen_axis=None):
    if chosen_axis is not None:
        axis = {'x': 0, 'y': 1, 'z': 2}[chosen_axis]
        y1, y2 = rot1[..., axis], rot2[..., axis]  # [Bs, 3]
        diff = torch.sum(y1 * y2, dim=-1)  # [Bs]
        diff = torch.clamp(diff, min=-1.0, max=1.0)
        return torch.acos(diff)
    else:
        mat_diff = torch.matmul(rot1, rot2.transpose(-1, -2))
        diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
        diff = (diff - 1) / 2.0
        diff = torch.clamp(diff, min=-1.0, max=1.0)
        return torch.acos(diff)


def rot_diff_degree(rot1, rot2, chosen_axis=None):
    return rot_diff_rad(rot1, rot2, chosen_axis=chosen_axis) / np.pi * 180.0


def rotate_about_axis(theta, axis='x'):
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])

    elif axis == 'y':
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])

    elif axis == 'z':
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta),  0],
                      [0, 0, 1]])
    return R

def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def chamfer_cpu(a, b, return_raw=False):  # [B, N, 3], only consider symmetrical cases
    dis = torch.norm(a.unsqueeze(1) - b.unsqueeze(2), dim=-1)  # [B, 1, N, 3] - [B, M, 1, 3] = [B, M, N, 3]
    mdis_a = torch.min(dis, dim=-2)[0]  # [B, M -> (), N] -> [B, N]
    mdis_b = torch.min(dis, dim=-1)[0]
    if return_raw:
        return mdis_a, mdis_b
    else:
        return torch.mean(mdis_a) + torch.mean(mdis_b)


class Basic_Utils():
    def __init__(self, config, chamfer_dist=chamfer_cpu):
        self.config = config
        self.chamfer_dist = chamfer_dist
        self.ycb_root = config.DATASET.data_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cur_path = os.path.abspath(os.path.dirname(__file__))
        config_path = '/'.join(cur_path.split('/')[:-1] + ['config/datasets/ycb_config'])
        with open(pjoin(config_path, 'classes.txt'), 'r') as f:
            self.ycb_cls_lst = [line for line in [line.strip() for line in f.readlines()] if len(line)]
            print(self.ycb_cls_lst)

        with open(pjoin(config_path, 'classes_info.yaml'), 'r') as f:
            self.classes_info = yaml.load(f, Loader=yaml.FullLoader)

        self.ycb_cls_ptsxyz_dict = {}
        self.ycb_cls_ptsxyz_cuda_dict = {}

        for cls in self.ycb_cls_lst:
            ptxyz_ptn = os.path.join(self.ycb_root, 'models', '{}/points.xyz'.format(cls))
            pointxyz = np.loadtxt(ptxyz_ptn.format(cls), dtype=np.float32)
            self.ycb_cls_ptsxyz_dict[cls] = pointxyz
            ptsxyz_cu = torch.from_numpy(pointxyz.astype(np.float32)).to(self.device)
            self.ycb_cls_ptsxyz_cuda_dict[cls] = ptsxyz_cu

    def get_cls_name(self, cls):
        if type(cls) is int:
            cls = self.ycb_cls_lst[cls - 1]
        return cls

    def mean_shift(self, data, radius=5.0):
        clusters = []
        for i in range(len(data)):
            cluster_centroid = data[i]
            cluster_frequency = np.zeros(len(data))
            # Search points in circle
            while True:
                temp_data = []
                for j in range(len(data)):
                    v = data[j]
                    # Handle points in the circles
                    if np.linalg.norm(v - cluster_centroid) <= radius:
                        temp_data.append(v)
                        cluster_frequency[i] += 1
                # Update centroid
                old_centroid = cluster_centroid
                new_centroid = np.average(temp_data, axis=0)
                cluster_centroid = new_centroid
                # Find the mode
                if np.array_equal(new_centroid, old_centroid):
                    break
            # Combined 'same' clusters
            has_same_cluster = False
            for cluster in clusters:
                if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= radius:
                    has_same_cluster = True
                    cluster['frequency'] = cluster['frequency'] + cluster_frequency
                    break
            if not has_same_cluster:
                clusters.append({
                    'centroid': cluster_centroid,
                    'frequency': cluster_frequency
                })

        print('clusters (', len(clusters), '): ', clusters)
        self.clustering(data, clusters)
        return clusters

    # Clustering data using frequency
    def clustering(self, data, clusters):
        t = []
        for cluster in clusters:
            cluster['data'] = []
            t.append(cluster['frequency'])
        t = np.array(t)
        # Clustering
        for i in range(len(data)):
            column_frequency = t[:, i]
            cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
            clusters[cluster_index]['data'].append(data[i])

    def cal_auc(self, add_dis, max_dis=0.1):
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf;
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
        aps = VOCap(D, acc)
        return aps * 100

    def cal_add_cuda(
        self, pred_RT, gt_RT, cls
    ):
        cls = self.get_cls_name(cls)
        p3ds = self.ycb_cls_ptsxyz_cuda_dict[cls]
        pred_p3ds = torch.matmul(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        gt_p3ds = torch.matmul(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
        return torch.mean(dis)

    def cal_adds_cuda(
        self, pred_RT, gt_RT, cls
    ):
        cls = self.get_cls_name(cls)
        p3ds = self.ycb_cls_ptsxyz_cuda_dict[cls]
        N, _ = p3ds.size()
        pd = torch.matmul(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        pd = pd.view(1, N, 3).repeat(N, 1, 1)
        gt = torch.matmul(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        gt = gt.view(N, 1, 3).repeat(1, N, 1)
        dis = torch.norm(pd - gt, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        return torch.mean(mdis)

    def cal_add_and_adds_batch(self, pred_RT, gt_RT, cls):  # [B, 3, 4]
        cls = self.get_cls_name(cls)
        p3ds = self.ycb_cls_ptsxyz_cuda_dict[cls].unsqueeze(0)  # [1, N, 3]
        pred_p3ds = torch.matmul(p3ds, pred_RT[..., :3].permute(0, 2, 1)) + pred_RT[..., 3:4].permute(0, 2, 1)
        gt_p3ds = torch.matmul(p3ds, gt_RT[..., :3].permute(0, 2, 1)) + gt_RT[..., 3:4].permute(0, 2, 1)
        dis = torch.norm(pred_p3ds - gt_p3ds, dim=-1).mean(dim=-1)  # [B, N, 3] -> [B]
        mdis, _ = self.chamfer_dist(pred_p3ds, gt_p3ds, return_raw=True)  # [B, N]
        mdis = mdis.mean(dim=-1)  # [B]
        return dis, mdis

    def cal_pose_error(self, pred_RT, gt_RT, cls):
        cls = self.get_cls_name(cls)
        sym_info = self.classes_info[cls]

        t_err = torch.norm(pred_RT[..., 3] - gt_RT[..., 3], dim=-1)  # [B]
        pred_rot, gt_rot = pred_RT[..., :3], gt_RT[..., :3]

        all_rmats = [np.eye(3)]
        chosen_axis = None

        for key, M in sym_info.items():
            if M > 10:
                chosen_axis = key
                continue
            next_rmats = []
            for k in range(M):
                rmat = rotate_about_axis(2 * np.pi * k / M, axis=key)
                for old_rmat in all_rmats:
                    next_rmats.append(np.matmul(rmat, old_rmat))
            all_rmats = next_rmats

        rmats = torch.from_numpy(np.array(all_rmats).astype(np.float32)).to(pred_rot.device)  # [M, 3, 3]
        gt_rots = torch.matmul(gt_rot.unsqueeze(1), rmats.unsqueeze(0))  # [B, M, 3, 3]
        rot_err = rot_diff_degree(gt_rots, pred_rot.unsqueeze(1), chosen_axis=chosen_axis)  # [B, M]
        rot_err, rot_idx = torch.min(rot_err, dim=-1)  # [B], [B]
        gt_rot = gt_rots[torch.tensor(np.arange(len(gt_rots))).to(rot_idx.device), rot_idx]  # [B, 3, 3]
        new_gt_RT = torch.cat([gt_rot, gt_RT[..., 3:4]], dim=-1)  # [B, 3, 4]

        return rot_err, t_err, new_gt_RT

    def cal_full_error(self, pred_RT, gt_RT, cls):
        r_err, t_err, new_gt_RT = self.cal_pose_error(pred_RT, gt_RT, cls)
        add, adds = self.cal_add_and_adds_batch(pred_RT, new_gt_RT, cls)
        add_acc, adds_acc = torch.mean((add <= 0.1).float()), torch.mean((adds <= 0.1).float())

        return {'add': add, 'adds': adds,
                'rdiff': r_err, 'tdiff': t_err,
                'add_acc': add_acc, 'adds_acc': adds_acc}  # all of shape [B]


