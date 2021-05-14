#!/usr/bin/env python3
import os
import numpy as np
import cv2
import random
import torch


class Basic_Utils():
    def __init__(self, config):
        self.config = config
        self.ycb_cls_lst = config.ycb_cls_lst
        self.ycb_cls_ptsxyz_dict = {}
        self.ycb_cls_ptsxyz_cuda_dict = {}

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


    def get_cls_name(self, cls):
        if type(cls) is int:
            cls = self.ycb_cls_lst[cls - 1]
        return cls

    def get_pointxyz(self, cls):
        cls = self.get_cls_name(cls)
        if cls in self.ycb_cls_ptsxyz_dict.keys():
            return self.ycb_cls_ptsxyz_dict[cls]
        ptxyz_ptn = os.path.join(
            self.config.ycb_root, 'models',
            '{}/points.xyz'.format(cls),
        )
        pointxyz = np.loadtxt(ptxyz_ptn.format(cls), dtype=np.float32)
        self.ycb_cls_ptsxyz_dict[cls] = pointxyz
        return pointxyz


    def get_pointxyz_cuda(self, cls):
        if cls in self.ycb_cls_ptsxyz_cuda_dict.keys():
            return self.ycb_cls_ptsxyz_cuda_dict[cls].clone()
        ptsxyz = self.get_pointxyz(cls)
        ptsxyz_cu = torch.from_numpy(ptsxyz.astype(np.float32)).cuda()
        self.ycb_cls_ptsxyz_cuda_dict[cls] = ptsxyz_cu
        return ptsxyz_cu.clone()


    def cal_add_cuda(
        self, pred_RT, gt_RT, p3ds
    ):
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
        return torch.mean(dis)

    def cal_adds_cuda(
        self, pred_RT, gt_RT, p3ds
    ):
        N, _ = p3ds.size()
        pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        pd = pd.view(1, N, 3).repeat(N, 1, 1)
        gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        gt = gt.view(N, 1, 3).repeat(1, N, 1)
        dis = torch.norm(pd - gt, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        return torch.mean(mdis)



def eval_metric(config, cls, gt_R, gt_T, pred_R, pred_T):

        mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1]).clone()
        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)
