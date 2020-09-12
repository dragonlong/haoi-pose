import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
from kaolin.nnsearch import nnsearch
import kaolin.cuda.sided_distance as sd
from scipy.spatial import cKDTree as Tree

DIVISION_EPS = 1e-10
SQRT_EPS = 1e-10
LS_L2_REGULARIZER = 1e-8

def breakpoint():
    import pdb;pdb.set_trace()
def hungarian_matching(cost, n_instance_gt):
    # cost is BxNxM
    B, N, M = cost.shape
    matching_indices = np.zeros([B, N], dtype=np.int32)
    for b in range(B):
        # limit to first n_instance_gt[b]
        _, matching_indices[b, :n_instance_gt[b]] = linear_sum_assignment(cost[b, :n_instance_gt[b], :])
    return matching_indices

def compute_miou_loss(W, W_gt, matching_indices=None):
    # W:     BxKxN, after softmax
    # W_gt - BxN, or B, K, N
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
    """

    if MULTI_HEAD:
        loss_nocs   = 0
        nocs_splits = torch.split(nocs, 3, dim=1)
        mask_splits = torch.split(mask_array, 1, dim=1)
        for i in range(num_parts):
            diff_l2 = torch.norm(nocs_splits[i] - nocs_gt, dim=1) # BxN
            diff_abs= torch.sum(torch.abs(nocs_splits[i] - nocs_gt), dim=1) # B*N
            if TYPE_LOSS=='L2':
                loss_nocs += torch.mean(mask_splits[i][:, 0, :]  * diff_l2, dim=1)
    else:
        diff_l2 = torch.norm(nocs - nocs_gt, dim=1) # BxN
        diff_abs= torch.sum(torch.abs(nocs - nocs_gt), dim=1) # B*N
        if TYPE_LOSS=='L2':
            loss_nocs = torch.mean(diff_l2, dim=1)

    return loss_nocs


def compute_vect_loss(vect, vect_gt, confidence=None, num_parts=2, mask_array=None, \
        TYPE_LOSS='L2'):
        """
        vect   : [B, K, N]; K could be any number, we only need the confidence
        vect_gt: [B, K, N];
        confidence: [B, N]
        """
        if confidence is not None:
            diff_l2 = torch.norm(vect - vect_gt, dim=1) * confidence # BxN
            diff_abs= torch.sum(torch.abs(vect - vect_gt), dim=1) * confidence # BxN
        else:
            diff_l2 = torch.norm(vect - vect_gt, dim=1) # BxN
            diff_abs= torch.sum(torch.abs(vect - vect_gt), dim=1) # BxN

        if TYPE_LOSS=='L2':
            if confidence is not None:
                return torch.mean(diff_l2 * confidence, dim=1)
            else:
                return torch.mean(diff_l2, dim=1)

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
    def __init__(self, gamma=2., weight=None, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


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
