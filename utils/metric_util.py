# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Utility functions for metric evaluation.

Author: Or Litany and Charles R. Qi
"""

import os
import sys
import torch
import logging
import open3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np

# Mesh IO
import trimesh
import __init__
from utils.extensions.chamfer_dist import ChamferDistance

class Metrics(object):
    ITEMS = [{
        'name': 'F-Score',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'ChamferDistance',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distance',
        'eval_object': ChamferDistance(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }]

    @classmethod
    def get(cls, pred, gt):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            _values[i] = eval_func(pred, gt)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):
        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        pred = cls._get_open3d_ptcloud(pred)
        gt = cls._get_open3d_ptcloud(gt)

        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)

        recall = float(sum(d < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d < th for d in dist1)) / float(len(dist1))
        return 2 * recall * precision / (recall + precision) if recall + precision else 0

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distance(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value

# ----------------------------------------
# Precision and Recall
# ----------------------------------------

def multi_scene_precision_recall(labels, pred, iou_thresh, conf_thresh, label_mask, pred_mask=None):
    '''
    Args:
        labels: (B, N, 6)
        pred: (B, M, 6)
        iou_thresh: scalar
        conf_thresh: scalar
        label_mask: (B, N,) with values in 0 or 1 to indicate which GT boxes to consider.
        pred_mask: (B, M,) with values in 0 or 1 to indicate which PRED boxes to consider.
    Returns:
        TP,FP,FN,Precision,Recall
    '''
    # Make sure the masks are not Torch tensor, otherwise the mask==1 returns uint8 array instead
    # of True/False array as in numpy
    assert(not torch.is_tensor(label_mask))
    assert(not torch.is_tensor(pred_mask))
    TP, FP, FN = 0, 0, 0
    if label_mask is None: label_mask = np.ones((labels.shape[0], labels.shape[1]))
    if pred_mask is None: pred_mask = np.ones((pred.shape[0], pred.shape[1]))
    for batch_idx in range(labels.shape[0]):
        TP_i, FP_i, FN_i = single_scene_precision_recall(labels[batch_idx, label_mask[batch_idx,:]==1, :],
                                                         pred[batch_idx, pred_mask[batch_idx,:]==1, :],
                                                         iou_thresh, conf_thresh)
        TP += TP_i
        FP += FP_i
        FN += FN_i

    return TP, FP, FN, precision_recall(TP, FP, FN)


def single_scene_precision_recall(labels, pred, iou_thresh, conf_thresh):
    """Compute P and R for predicted bounding boxes. Ignores classes!
    Args:
        labels: (N x bbox) ground-truth bounding boxes (6 dims)
        pred: (M x (bbox + conf)) predicted bboxes with confidence and maybe classification
    Returns:
        TP, FP, FN
    """


    # for each pred box with high conf (C), compute IoU with all gt boxes.
    # TP = number of times IoU > th ; FP = C - TP
    # FN - number of scene objects without good match

    gt_bboxes = labels[:, :6]

    num_scene_bboxes = gt_bboxes.shape[0]
    conf = pred[:, 6]

    conf_pred_bbox = pred[np.where(conf > conf_thresh)[0], :6]
    num_conf_pred_bboxes = conf_pred_bbox.shape[0]

    # init an array to keep iou between generated and scene bboxes
    iou_arr = np.zeros([num_conf_pred_bboxes, num_scene_bboxes])
    for g_idx in range(num_conf_pred_bboxes):
        for s_idx in range(num_scene_bboxes):
            iou_arr[g_idx, s_idx] = calc_iou(conf_pred_bbox[g_idx ,:], gt_bboxes[s_idx, :])


    good_match_arr = (iou_arr >= iou_thresh)

    TP = good_match_arr.any(axis=1).sum()
    FP = num_conf_pred_bboxes - TP
    FN = num_scene_bboxes - good_match_arr.any(axis=0).sum()

    return TP, FP, FN


def precision_recall(TP, FP, FN):
    Prec = 1.0 * TP / (TP + FP) if TP+FP>0 else 0
    Rec = 1.0 * TP / (TP + FN)
    return Prec, Rec


def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """

    max_a = box_a[0:3] + box_a[3:6]/2
    max_b = box_b[0:3] + box_b[3:6]/2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6]/2
    min_b = box_b[0:3] - box_b[3:6]/2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0*intersection / union


if __name__ == '__main__':
    print('running some tests')

    ############
    ## Test IoU
    ############
    box_a = np.array([0,0,0,1,1,1])
    box_b = np.array([0,0,0,2,2,2])
    expected_iou = 1.0/8
    pred_iou = calc_iou(box_a, box_b)
    assert expected_iou == pred_iou, 'function returned wrong IoU'

    box_a = np.array([0,0,0,1,1,1])
    box_b = np.array([10,10,10,2,2,2])
    expected_iou = 0.0
    pred_iou = calc_iou(box_a, box_b)
    assert expected_iou == pred_iou, 'function returned wrong IoU'

    print('IoU test -- PASSED')

    #########################
    ## Test Precition Recall
    #########################
    gt_boxes = np.array([[0,0,0,1,1,1],[3, 0, 1, 1, 10, 1]])
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0],[3, 0, 1, 1, 10, 1, 0.9]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 2 and FP == 0 and FN == 0
    assert precision_recall(TP, FP, FN) == (1, 1)

    detected_boxes = np.array([[0,0,0,1,1,1, 1.0]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 1 and FP == 0 and FN == 1
    assert precision_recall(TP, FP, FN) == (1, 0.5)

    detected_boxes = np.array([[0,0,0,1,1,1, 1.0], [-1,-1,0,0.1,0.1,1, 1.0]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 1 and FP == 1 and FN == 1
    assert precision_recall(TP, FP, FN) == (0.5, 0.5)

    # wrong box has low confidence
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0], [-1,-1,0,0.1,0.1,1, 0.1]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 1 and FP == 0 and FN == 1
    assert precision_recall(TP, FP, FN) == (1, 0.5)

    print('Precition Recall test -- PASSED')
