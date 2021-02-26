import os
import sys
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
from kaolin.nnsearch import nnsearch
import kaolin.cuda.sided_distance as sd
from scipy.spatial import cKDTree as Tree
from typing import Any
from collections import OrderedDict
DIVISION_EPS = 1e-10
SQRT_EPS = 1e-10
LS_L2_REGULARIZER = 1e-8
epsilon = 1e-10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'common'))
from common.debugger import *
from common.d3_utils import rotate_about_axis

# 0.94
FAR_THRESHOLD  = 0.3
NEAR_THRESHOLD = 0.1
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

# # 0.93
# FAR_THRESHOLD  = 0.3
# NEAR_THRESHOLD = 0.2
# GT_VOTE_FACTOR = 3 # number of GT votes per point
# OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness
#

# # 0.92
# FAR_THRESHOLD  = 0.4
# NEAR_THRESHOLD = 0.2
# GT_VOTE_FACTOR = 3 # number of GT votes per point
# OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

# # 0.9
# FAR_THRESHOLD  = 0.3
# NEAR_THRESHOLD = 0.1
# GT_VOTE_FACTOR = 3 # number of GT votes per point
# OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

# FAR_THRESHOLD  = 0.6
# NEAR_THRESHOLD = 0.3
# GT_VOTE_FACTOR = 3 # number of GT votes per point
# OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

import __init__
from global_info import global_info
from common.d3_utils import compute_rotation_matrix_from_ortho6d, compute_geodesic_distance_from_two_matrices
# global variables
infos           = global_info()
sym_type        = infos.sym_type
"""
loss_geodesic(predict_rmat, gt_rmat): [B, 3, 3]
FocalLoss(nn.Module):
compute_miou_loss(W, W_gt)
compute_vote_loss(end_points): voting loss
compute_nocs_loss(nocs, nocs_gt, confidence)
compute_vect_loss(vect, vect_gt): L2/L1 between two [B, K, N] vectors
compute_multi_offsets_loss(vect, vect_gt, confidence=None): K=63

chamfer_distance(S1: torch.Tensor, S2: torch.Tensor):
directed_distance(S1: torch.Tensor, S2: torch.Tensor):

compute_objectness_loss(end_points): Compute objectness loss for the proposals.
compute_box_and_sem_cls_loss(end_points, config=None): Compute 3D bounding box and semantic classification loss.
"""

def bp():
    import pdb;pdb.set_trace()

def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
    return dist1, idx1, dist2, idx2

def hungarian_matching(cost, n_instance_gt):
    # cost is BxNxM
    B, N, M = cost.shape
    matching_indices = np.zeros([B, N], dtype=np.int32)
    for b in range(B):
        # limit to first n_instance_gt[b]
        _, matching_indices[b, :n_instance_gt[b]] = linear_sum_assignment(cost[b, :n_instance_gt[b], :])
    return matching_indices

def loss_geodesic(predict_rmat, gt_rmat):
    """
    gt_rmat: b*3*3
    predict_rmat: b*3*3
    """

    theta = compute_geodesic_distance_from_two_matrices(gt_rmat, predict_rmat)
    error = theta/np.pi * 180 # to degrees
    return error

def loss_vectors(v1, v2):
    """
    input: [B, 3, N]
    [B, 3, N]
    """
    r_diff = torch.acos( torch.sum(v1*v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + epsilon) ) * 180 / np.pi
    error = r_diff.mean()
    return error

def compute_reconstruction_loss(mano_pred, mano_gt, loss_type='L2'):
    """
    mano_pre: N, 45
    mano_gt: N, 45
    """
    reconstruct_loss = None
    return reconstruct_loss


def compute_miou_loss(W, W_gt, matching_indices=None, loss_type='miou'):
    # W:     BxKxN, after softmax
    # W_gt - BxN, or B, K, N
    if loss_type=='xentropy':
        return torch.nn.functional.nll_loss(torch.log(W), W_gt.long())
    else:
        C = W.size(1)
        if len(W_gt.size())<3:
            W_gt = F.one_hot(W_gt.long(), num_classes=C).permute(0, 2, 1).contiguous() # B, K, N
        dot   = torch.sum(W_gt * W, dim=2) # BxK
        denominator = torch.sum(W_gt, dim=2) + torch.sum(W, dim=2) - dot # BxK
        mIoU = dot / (denominator + DIVISION_EPS) # BxK
        return 1.0 - mIoU #

def compute_nocs_loss(nocs, nocs_gt, confidence, \
                        num_parts=2, mask_array=None, \
                        TYPE_LOSS='L2', MULTI_HEAD=False, \
                        SELF_SU=False):
    """
    nocs:       [B, 3K, N]
    nocs_gt:    [B, 3, N]
    confidence: [B, 1, N]
    mask_array: [B, K, N]


    return B
    """
    if MULTI_HEAD:
        loss_nocs   = 0
        nocs_splits = torch.split(nocs, 3, dim=1)
        mask_splits = torch.split(mask_array, 1, dim=1)
        for i in range(num_parts):
            diff_l2 = torch.norm(nocs_splits[i] - nocs_gt, dim=1) # BxN
            diff_abs= torch.sum(torch.abs(nocs_splits[i] - nocs_gt), dim=1) # B*N
            if TYPE_LOSS=='L2':
                loss_nocs += torch.sum(mask_splits[i][:, 0, :]  * diff_l2, dim=1)/(torch.sum(mask_splits[i][:, 0, :], dim=1) + 1)
    else:
        diff_l2 = torch.norm(nocs - nocs_gt, dim=1) # ([B, 3, N], [B, 3, N]) --> BxN
        diff_abs= torch.sum(torch.abs(nocs - nocs_gt), dim=1) # B*N
        if TYPE_LOSS=='L2':
            loss_nocs = torch.mean(diff_l2, dim=1)

    return loss_nocs

def compute_1vN_nocs_loss(nocs, nocs_gt, confidence=None, target_category='remote', class_array=None,\
                        num_parts=2, mask_array=None, \
                        TYPE_LOSS='L2',
                        SELF_SU=False):
    """
    return [B]
    nocs:       [B, 3K, N]
    nocs_gt:    [B, 3, N] -> [B, 3 * K, N], with basic Y matrix,
    confidence: [B, 1, N]
    mask_array: [B, K, N]

    if class_array is not None, then we map it to rotation interval;

    get possible M base, R= [3, 3]
    apply to GT, get M of them;
    from [B, 3, N] to [B, M, 3, N] --> B, M --> find the minimum loss out of M candidates per;
    sum together
    """
    loss_nocs   = 0
    nocs_splits = torch.split(nocs, 3, dim=1)
    if mask_array is not None:
        mask_splits = torch.split(mask_array, 1, dim=1)
    elif confidence is not None and num_parts==1:
        mask_splits = [confidence.unsqueeze(1)]
    else:
        mask_splits = None
    # augment GT with category infos
    all_rmats = []
    for key, M in sym_type[target_category].items():
        for k in range(M):
            rmat = rotate_about_axis(2 * np.pi * k / M, axis=key)
            all_rmats.append(rmat)
    # reshape all_rmats into tensor array,
    rmats = torch.from_numpy(np.array(all_rmats).astype(np.float32)).cuda()

    # we get the updated [M, 3, 3] * [B, 1, 3, N] --> B, M, 3, N
    nocs_gt_aug = torch.matmul(rmats, torch.unsqueeze(nocs_gt, 1)-0.5) + 0.5
    for i in range(num_parts): #
        diff_l2 = torch.norm(nocs_splits[i].unsqueeze(1) - nocs_gt_aug, dim=2) # BxMxN
        # diff_abs= torch.sum(torch.abs(nocs_splits[i] - nocs_gt), dim=1)        # BxMxN
        if mask_splits is not None:
            loss_part = torch.sum(mask_splits[i][:, 0:1, :]  * diff_l2, dim=-1)/(torch.sum(mask_splits[i][:, 0:1, :], dim=-1) + 1)   # [B, M, N] * [B, 1, N] -> B, M
        else:
            loss_part = torch.mean(diff_l2, dim=-1)
        loss_part, _ =  torch.min(loss_part, dim=-1)
        loss_nocs += loss_part

    return loss_nocs

def compute_vect_loss(vect, vect_gt, confidence=None, num_parts=2, mask_array=None, \
        TYPE_LOSS='L2'):
        """
        vect   : [B, K, N]; K could be any number, we only need the confidence
        vect_gt: [B, K, N];
        confidence: [B, N]
        """
        diff_l2 = torch.norm(vect - vect_gt, dim=1) # BxN
        diff_abs= torch.sum(torch.abs(vect - vect_gt), dim=1) # BxN
        diff_avg= torch.mean(torch.abs(vect - vect_gt), dim=1) # B*N
        if confidence is not None:
            if TYPE_LOSS=='L2':
                return torch.sum(diff_l2 * confidence, dim=1) / (torch.sum(confidence, dim=1) + 1)
            elif TYPE_LOSS=='L1':
                return torch.mean(diff_avg * confidence, dim=1)
            elif TYPE_LOSS=='SOFT_L1':
                return huber_loss(vect - vect_gt)
        else:
            if TYPE_LOSS=='L2':
                return torch.mean(diff_l2 * confidence , dim=1)
            elif TYPE_LOSS=='L1':
                return torch.mean(diff_avg, dim=1)
            elif TYPE_LOSS=='SOFT_L1':
                return huber_loss(vect - vect_gt)

def compute_multi_offsets_loss(vect, vect_gt, confidence=None):
    """
    vect: B, 63, N
    vect_gt: B, 63, N
    """
    diff_p  = (vect - vect_gt).view(vect.size(0), 21, -1, vect.size(-1)).contiguous()
    diff_l2 = torch.norm(diff_p, dim=2)

    return torch.mean(torch.mean(diff_l2, dim=1) * confidence, dim=1)

class SidedDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S1, S2):

        batchsize, n, _ = S1.size()
        S1 = S1.contiguous()
        S2 = S2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        dist1 = dist1.cuda()
        idx1 = idx1.cuda()
        try:
            sd.forward(S1, S2, dist1, idx1)
        except BaseException:
            sd.forward_cuda(S1, S2, dist1, idx1)

        return idx1.long()


class SidedDistance(torch.nn.Module):
    """
    For every point in set1 find the indecies of the closest point in set2
    Args:
            set1 (torch.cuda.Tensor) : set of pointclouds of shape B x N x 3
            set2 (torch.cuda.Tensor) : set of pointclouds of shape B x M x 3
    Returns:
            torch.cuda.Tensor: indecies of the closest points in set2
    Example:
            >>> A = torch.rand(2,300,3)
            >>> B = torch.rand(2,200,3)
            >>> sided_minimum_dist = SidedDistance()
            >>> indices = sided_minimum_dist(A,B)
            >>> indices.shape
            torch.Size([2, 300])
    """

    def forward(self, S1: torch.Tensor, S2: torch.Tensor):
        assert len(S1.shape) == 3
        assert len(S2.shape) == 3
        return SidedDistanceFunction.apply(S1, S2).detach()


def chamfer_distance(S1: torch.Tensor, S2: torch.Tensor,
                     w1: float = 1., w2: float = 1.):
    """
    Computes the chamfer distance between two point clouds
    Args:
            S1 (torch.Tensor): point cloud
            S2 (torch.Tensor): point cloud
            w1: (float): weighting of forward direction
            w2: (float): weighting of backward direction
    Returns:
            torch.Tensor: chamfer distance between two point clouds S1 and S2
    Example:
            >>> A = torch.rand(300,3)
            >>> B = torch.rand(200,3)
            >>> >>> chamfer_distance(A,B)
            tensor(0.1868)
    """

    assert (S1.dim() == S2.dim()), 'S1 and S2 must have the same dimesionality'
    assert (S1.dim() == 2), 'the dimensions of the input must be 2 '

    dist_to_S2 = directed_distance(S1, S2)
    dist_to_S1 = directed_distance(S2, S1)

    distance = w1 * dist_to_S2 + w2 * dist_to_S1

    return distance


def directed_distance(S1: torch.Tensor, S2: torch.Tensor, mean: bool = True):
    r"""Computes the average distance from point cloud S1 to point cloud S2
    Args:
            S1 (torch.Tensor): point cloud
            S2 (torch.Tensor): point cloud
            mean (bool): if the distances should be reduced to the average
    Returns:
            torch.Tensor: ditance from point cloud S1 to point cloud S2
    Args:
    Example:
            >>> A = torch.rand(300,3)
            >>> B = torch.rand(200,3)
            >>> >>> directed_distance(A,B)
            tensor(0.1868)
    """

    if S1.is_cuda and S2.is_cuda:
        sided_minimum_dist = SidedDistance()
        closest_index_in_S2 = sided_minimum_dist(
            S1.unsqueeze(0), S2.unsqueeze(0))[0]
        closest_S2 = torch.index_select(S2, 0, closest_index_in_S2)

    else:
        from time import time
        closest_index_in_S2 = nnsearch(S1, S2)
        closest_S2 = S2[closest_index_in_S2]

    dist_to_S2 = (((S1 - closest_S2)**2).sum(dim=-1))
    if mean:
        dist_to_S2 = dist_to_S2.mean()

    return dist_to_S2

def iou(points1: torch.Tensor, points2: torch.Tensor, thresh=.5):
    r""" Computes the intersection over union values for two sets of points
    Args:
            points1 (torch.Tensor): first points
            points2 (torch.Tensor): second points
    Returns:
            iou (torch.Tensor) : IoU scores for the two sets of points
    Examples:
            >>> points1 = torch.rand( 1000)
            >>> points2 = torch.rand( 1000)
            >>> loss = iou(points1, points2)
            tensor(0.3400)
    """
    points1[points1 <= thresh] = 0
    points1[points1 > thresh] = 1

    points2[points2 <= thresh] = 0
    points2[points2 > thresh] = 1

    points1 = points1.view(-1).byte()
    points2 = points2.view(-1).byte()

    assert points1.shape == points2.shape, 'points1 and points2 must have the same shape'

    iou = torch.sum(torch.mul(points1, points2).float()) / \
        torch.sum((points1 + points2).clamp(min=0, max=1).float())

    return iou

def f_score(gt_points: torch.Tensor, pred_points: torch.Tensor,
            radius: float = 0.01, extend=False):
    r""" Computes the f-score of two sets of points, with a hit defined by two point existing withing a defined radius of each other
    Args:
            gt_points (torch.Tensor): ground truth points
            pred_points (torch.Tensor): predicted points points
            radius (float): radisu from a point to define a hit
            extend (bool): if the alternate f-score definition should be applied
    Returns:
            (float): computed f-score
    Example:
            >>> points1 = torch.rand(1000)
            >>> points2 = torch.rand(1000)
            >>> loss = f_score(points1, points2)
            >>> loss
            tensor(0.0070)
    """

    pred_distances = torch.sqrt(directed_distance(
        gt_points, pred_points, mean=False))
    gt_distances = torch.sqrt(directed_distance(
        pred_points, gt_points, mean=False))

    if extend:
        fp = (gt_distances > radius).float().sum()
        tp = (gt_distances <= radius).float().sum()
        precision = tp / (tp + fp)
        tp = (pred_distances <= radius).float().sum()
        fn = (pred_distances > radius).float().sum()
        recall = tp / (tp + fn)

    else:
        fn = torch.sum(pred_distances > radius)
        fp = torch.sum(gt_distances > radius).float()
        tp = torch.sum(gt_distances <= radius).float()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

    f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f_score

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2., weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)

class FocalLoss(nn.Module):
    def __init__(
        self, gamma: float = 2, alphas: Any = None, size_average: bool = True, normalized: bool = True,
    ):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._alphas = alphas
        self.size_average = size_average
        self.normalized = normalized

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = torch.gather(logpt, -1, target.unsqueeze(-1))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self._alphas is not None:
            at = self._alphas.gather(0, target)
            logpt = logpt * Variable(at)

        if self.normalized:
            sum_ = 1 / torch.sum((1 - pt) ** self._gamma)
        else:
            sum_ = 1

        loss = -1 * sum_ * (1 - pt) ** self._gamma * logpt
        return loss.sum()

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2., weight=None, ignore_index=-1):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.nll_loss = nn.NLLLoss(weight, ignore_index)
#
#     def forward(self, inputs, targets):
#         return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>. loss for VoteNet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
# # try to sum up all related loss
# def get_loss(end_points, config):
#     """ Loss functions
#
#     Args:
#         end_points: dict
#             {
#                 seed_xyz, seed_inds,
#                 center,
#                 heading_scores, heading_residuals_normalized,
#                 size_scores, size_residuals_normalized,
#                 sem_cls_scores, #seed_logits,#
#                 center_label,
#                 heading_class_label, heading_residual_label,
#                 size_class_label, size_residual_label,
#                 sem_cls_label,
#                 box_label_mask,
#                 vote_label, vote_label_mask
#             }
#         config: dataset config instance
#     Returns:
#         loss: pytorch scalar tensor
#         end_points: dict
#     """
#
#     # Obj loss
#     objectness_loss, objectness_label, objectness_mask, object_assignment = \
#         compute_objectness_loss(end_points)
#     end_points['objectness_loss'] = objectness_loss
#     end_points['objectness_label'] = objectness_label
#     end_points['objectness_mask'] = objectness_mask
#     end_points['object_assignment'] = object_assignment
#     total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
#     end_points['pos_ratio'] = \
#         torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
#     end_points['neg_ratio'] = \
#         torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']
#
#     # Box loss and sem cls loss
#     center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
#         compute_box_and_sem_cls_loss(end_points, config)
#     end_points['center_loss'] = center_loss
#     end_points['heading_cls_loss'] = heading_cls_loss
#     end_points['heading_reg_loss'] = heading_reg_loss
#     end_points['size_cls_loss'] = size_cls_loss
#     end_points['size_reg_loss'] = size_reg_loss
#     end_points['sem_cls_loss'] = sem_cls_loss
#     box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
#     end_points['box_loss'] = box_loss
#
#     # Final loss function
#     loss = 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
#     loss *= 10
#     end_points['loss'] = loss
#
#     # --------------------------------------------
#     # Some other statistics
#     obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
#     obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
#     end_points['obj_acc'] = obj_acc
#
#     return loss, end_points


def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size= end_points['seed_xyz'].shape[0]
    num_seed  = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz  = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0] # loss
    K = aggregated_vote_xyz.shape[1] #
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask : 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1  = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config=None):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    # num_heading_bin = config.num_heading_bin
    # num_size_cluster = config.num_size_cluster
    # num_class = config.num_class
    # mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask   = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6) # on every predicted center;

    centroid_reg_loss2 = torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6) # on every gt center

    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # compute the heat-map confidence loss
    thres_r = NEAR_THRESHOLD
    gt_confidence_score = 1 -  dist1.detach()/ thres_r
    gt_confidence_score[gt_confidence_score<0] = 0

    # for those considered as object, we further add center confidence
    # distance_regression_loss = 0
    distance_regression_loss = torch.sum( objectness_label * huber_loss(end_points['center_confidence'] - gt_confidence_score)) / (torch.sum(objectness_label)+1e-6) # B, K
    # # Compute heading loss
    # heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    # criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    # heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    # heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
    #
    # heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    # heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)
    #
    # # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    # heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    # heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    # heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    # heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)
    #
    # # Compute size loss
    # size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    # criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    # size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    # size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
    #
    # size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    # size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    # size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    # size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    # predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)
    #
    # mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3)
    # mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    # size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    # size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    # size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)
    #
    # # 3.4 Semantic cls loss
    # sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    # criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    # sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    # sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, centroid_reg_loss1, centroid_reg_loss2, distance_regression_loss#, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def get_loss(end_points, config=None):
    """ Loss functions

    Args:
        end_points: dict
            {
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """
    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    # , heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss
    center_loss, centroid_reg_loss1, centroid_reg_loss2, confidence_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['center_loss1'] = centroid_reg_loss1
    end_points['center_loss2'] = centroid_reg_loss2
    end_points['confidence_loss'] = confidence_loss
    # end_points['heading_cls_loss'] = heading_cls_loss
    # end_points['heading_reg_loss'] = heading_reg_loss
    # end_points['size_cls_loss'] = size_cls_loss
    # end_points['size_reg_loss'] = size_reg_loss
    # end_points['sem_cls_loss'] = sem_cls_loss
    # box_loss = 0.5 * center_loss + confidence_loss # + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    # end_points['box_loss'] = box_loss

    box_loss = 0.2 * center_loss + confidence_loss

    # Final loss function
    # first 0.1, later 0.5,
    loss = vote_loss + 0.5*objectness_loss + config.TRAIN.confidence_loss_multiplier * box_loss # + 0.1*sem_cls_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc
    print('Acc is ', end_points['obj_acc'].cpu().numpy())

    return loss, end_points


if __name__ == "__main__":
    # FL = FocalLoss()
    # CE = nn.CrossEntropyLoss()
    # N = 4
    # C = 20
    # S = 14
    # inputs = torch.randn(N, C, 14, 480, 480)
    # targets = torch.empty(N, 14, 480, 480, dtype=torch.long).random_(0, C)
    # print('----inputs----')
    # print(inputs)
    # print('---target-----')
    # print(targets)
    #
    # fl_loss = FL(inputs, targets)
    # ce_loss = CE(inputs, targets)
    # print('ce = {}, fl ={}'.format(ce_loss, fl_loss))


    B = 4
    C = 5
    N = 1000
    inputs = torch.randn(B, C, N)
    targets = torch.empty(B, N, dtype=torch.long).random_(0, C)
    loss = compute_miou_loss(inputs, targets)
    print('loss is: ', loss.shape, '\n', loss)
