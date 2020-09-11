import torch 
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.optimize import linear_sum_assignment
DIVISION_EPS = 1e-10
SQRT_EPS = 1e-10
LS_L2_REGULARIZER = 1e-8

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

def hungarian_matching(cost, n_instance_gt):
    # cost is BxNxM
    B, N, M = cost.shape
    matching_indices = np.zeros([B, N], dtype=np.int32)
    for b in range(B):
        # limit to first n_instance_gt[b]
        _, matching_indices[b, :n_instance_gt[b]] = linear_sum_assignment(cost[b, :n_instance_gt[b], :])
    return matching_indices

def compute_nocs_loss(nocs, nocs_gt, confidence, \
                        num_parts=2, mask_array=None, \
                        TYPE_L='L2', MULTI_HEAD=False, \
                        SELF_SU=False):
    """
    nocs:     [B, 3K, N]
    nocs_gt:  [B, 3, N]
    confidence: [B, 1, N]
    mask_array: [B, K, N]
    """

    if MULTI_HEAD:
        loss_nocs   = 0
        nocs_splits = torch.split(nocs, split_size_or_sections=num_parts, dim=1)
        mask_splits = torch.split(mask_array, split_size_or_sections=num_parts, dim=1)
        for i in range(num_parts):
            diff_l2 = torch.norm(nocs_splits[i] - nocs_gt, dim=1) # BxN
            diff_abs= torch.sum(torch.abs(nocs_splits[i] - nocs_gt), dim=1) # B*N
            if TYPE_L=='L2':
                loss_nocs += torch.mean(mask_splits[i][:, 0, :]  * diff_l2, dim=1)
    else:
        diff_l2 = torch.norm(nocs - nocs_gt, dim=1) # BxN
        diff_abs= torch.sum(torch.abs(nocs - nocs_gt), dim=1) # B*N
        if TYPE_L=='L2':
            loss_nocs = torch.mean(diff_l2, dim=1)

    return loss_nocs

def compute_miou_loss(W, W_gt, matching_indices=None):
    # W - BxKxN
    # I_gt - BxN
    W_reordered = W
    depth = tf.shape(W)[2]
    dot   = tf.reduce_sum(W_gt * W_reordered, axis=1) # BxK
    denominator = torch.sum(W_gt, axis=1) + torch.sum(W_reordered, axis=1) - dot
    mIoU = dot / (denominator + DIVISION_EPS) # BxK
    return 1.0 - mIoU

def compute_vect_loss(vect, vect_gt, confidence=2, num_parts=2, mask_array=None, \
        TYPE_LOSS='L2', MULTI_HEAD=False):
        if confidence is not None:
            diff_l2 = torch.norm(vect - vect_gt, dim=1) * confidence # BxN
            diff_abs= torch.sum(tf.abs(vect - vect_gt), dim=1) * confidence # BxN
        else:
            diff_l2 = torch.norm(vect - vect_gt, dim=1) # BxN
            diff_abs= torch.sum(tf.abs(vect - vect_gt), dim=1) # BxN

        if TYPE_LOSS=='L2':
            return torch.mean(diff_l2, dim=1) 

# def smooth_l1_loss(y_true, y_pred):
#     """Implements Smooth-L1 loss.
#     y_true and y_pred are typicallly: [N, 4], but could be any shape.
#     """
#     diff = K.abs(y_true - y_pred)
#     less_than_one = K.cast(K.less(diff, 1.0), "float32")
#     loss = (less_than_one * 0.5 * diff**2) + (1-less_than_one) * (diff - 0.5)

#     return loss

# def smooth_l1_diff(diff, threshold = 0.1):
#     coefficient = 1 / (2 * threshold)
#     #coefficient = tf.Print(coefficient, [coefficient], message='coefficient', summarize=15)

#     less_than_threshold = K.cast(K.less(diff, threshold), "float32")
#     #less_than_threshold = tf.Print(less_than_threshold, [less_than_threshold], message='less_than_threshold', summarize=15)

#     loss = (less_than_threshold * coefficient * diff ** 2) + (1 - less_than_threshold) * (diff - threshold / 2)

#     return loss

if __name__ == "__main__":
    FL = FocalLoss()
    CE = nn.CrossEntropyLoss()
    N = 4
    C = 20
    S = 14
    inputs = torch.randn(N, C, 14, 480, 480)
    targets = torch.empty(N, 14, 480, 480, dtype=torch.long).random_(0, C)
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss = FL(inputs, targets)
    ce_loss = CE(inputs, targets)
    print('ce = {}, fl ={}'.format(ce_loss, fl_loss))