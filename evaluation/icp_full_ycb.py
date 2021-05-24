# examples/Python/Advanced/global_registration.py

import torch
import open3d as o3d
import numpy as np
import copy
import hydra
import copy
import os
import sys
import matplotlib.pyplot as plt
from os.path import join as pjoin
cur_path = os.path.dirname(__file__)
sys.path.insert(0, pjoin(cur_path, '../'))
from dataset.ycb_dataset import YCBDataset
from utils.ycb_eval_utils import Basic_Utils
import json

anchors = np.load(pjoin(cur_path, '..', 'vgtk', 'data', 'anchors', 'anchors.npy'))


def rot_diff_rad(rot1, rot2, chosen_axis=None):
    if chosen_axis is not None:
        axis = {'x': 0, 'y': 1, 'z': 2}[chosen_axis]
        y1, y2 = rot1[..., axis], rot2[..., axis]  # [Bs, 3]
        diff = np.sum(y1 * y2, axis=-1)  # [Bs]
    else:
        mat_diff = np.matmul(rot1, rot2.swapaxes(-1, -2))
        diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
        diff = (diff - 1) / 2.0
    diff = np.clip(diff, a_min=-1.0, a_max=1.0)
    return np.arccos(diff)


def rot_diff_degree(rot1, rot2, chosen_axis=None):
    return rot_diff_rad(rot1, rot2, chosen_axis=chosen_axis) / np.pi * 180.0


def bp():
    import pdb;pdb.set_trace()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def draw_geo(geo_list):
    """
    pcd_list = []
    for geo in geo_list:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(geo)
        pcd_list.append(pcd)

    o3d.visualization.draw_geometries(pcd_list)
    """
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    for geo in geo_list:
        ax.scatter(geo[..., 0], geo[..., 1], geo[..., 2])
    plt.show()


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def base_icp(source, target, init_trans, method='point2plane', threshold=0.05, max_iter=2000):
    """threshold: assume inputs are normalized; init may have a large translation error"""
    method_dict = {'point2plane': o3d.pipelines.registration.TransformationEstimationPointToPlane,
                   'point2point': o3d.pipelines.registration.TransformationEstimationPointToPoint}
    method = method_dict[method]()
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_trans, method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))


def eval_registeration(partial, full, trans):
    """
    set threshold to 1.0 -> want to compute the rmse for all partial points
    note that evaluate_registration will find correspondences for the *first* argument
    """
    result = o3d.pipelines.registration.evaluate_registration(partial, full,
                                                              1.0, trans)
    return result.inlier_rmse


def get_init_pose(full, partial, method='rotate_60', voxel_size=0.05):
    if method == 'global':
        source_down, source_fpfh = preprocess_point_cloud(partial, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(full, voxel_size)
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        init_poses = np.expand_dims(np.asarray(result_ransac.transformation), 0)
    elif method == 'rotate_60':
        def get_bbox_center(pts):  # pts: [...., 3, N]
            return (np.min(pts, axis=-1, keepdims=True) +
                    np.max(pts, axis=-1, keepdims=True)) * 0.5

        partial = np.asarray(partial.points)
        partial = partial.reshape((1, ) + partial.shape).transpose(0, 2, 1).repeat(
            len(anchors), 0)
        rotated_partial = torch.matmul(torch.tensor(anchors).float(),
                                       torch.tensor(partial).float()).numpy()
        centers = get_bbox_center(rotated_partial)
        init_poses = np.zeros((len(anchors), 4, 4))
        init_poses[..., :3, :3] = anchors
        init_poses[..., :3, 3:] = -centers
        init_poses[..., 3, 3] = 1
    else:
        assert 0, f'Unsupported method {method}'

    return init_poses


def icp_registeration(full, partial, init_method='rotate_60', voxel_size=0.05,
                      icp_method='point2point', icp_threshold=0.05, icp_max_iter=2000):
    init_poses = get_init_pose(copy.deepcopy(full), copy.deepcopy(partial),
                               method=init_method, voxel_size=voxel_size)
    best_pose, min_rmse = None, 1e9
    for init_pose in init_poses:
        cur_result = base_icp(copy.deepcopy(partial), copy.deepcopy(full), init_pose,
                              method=icp_method, threshold=icp_threshold, max_iter=icp_max_iter)
        cur_rmse = eval_registeration(partial, full, cur_result.transformation)
        if cur_rmse < min_rmse:
            min_rmse = cur_rmse
            best_pose = cur_result.transformation
    return best_pose


def eval_pose(pose, r, t):
    rdiff = rot_diff_degree(pose[:3, :3], r)
    tdiff = pose[:3, 3] - t
    tdiff = np.sqrt((tdiff ** 2).sum())

    return {'rdiff': rdiff, 'tdiff': tdiff,
            '5deg': float(rdiff <= 5), '5cm': float(tdiff <= 0.05),
            '5deg5cm': float(rdiff <= 5 and tdiff <= 0.05)}


@hydra.main(config_path="../config/icp.yaml")
def main(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    dataset = YCBDataset(cfg, cfg.DATASET.data_path, 'val')
    bs = Basic_Utils(cfg)
    err_dict = {'global': {}, 'rotate_60': {}}
    for i in range(len(dataset)):
        data_dict = dataset[i]
        partial, full = data_dict['xyz'], data_dict['full']
        r, t = data_dict['R_gt'].numpy(), data_dict['T'].numpy()  # already normalized
        gt_pose = np.eye(4)
        gt_pose[:3, :3], gt_pose[:3, 3] = r, t
        full = full - 0.5
        partial_pts, full_pts = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        partial_pts.points = o3d.utility.Vector3dVector(partial)
        full_pts.points = o3d.utility.Vector3dVector(full)
        for init_method in ['global', 'rotate_60']:
            pose = icp_registeration(copy.deepcopy(full_pts),
                                     copy.deepcopy(partial_pts),
                                     init_method=init_method)
            # draw_registration_result(partial_pts, full_pts, pose)
            pred_pose = np.linalg.inv(pose)
            # print(eval_pose(real_pose, r, t))
            err = bs.cal_full_error(torch.tensor(pred_pose).float().to(bs.device).unsqueeze(0),
                                    torch.tensor(gt_pose).float().to(bs.device).unsqueeze(0), cfg.instance)
            cur_dict = err_dict[init_method]
            for key, value in err.items():
                if key not in cur_dict:
                    cur_dict[key] = []
                cur_dict[key].append(float(value.cpu().numpy()))

    final_dict = {}
    for init_method in ['global', 'rotate_60']:
        cur_dict = err_dict[init_method]
        final_dict[init_method] = {}
        cur_final = final_dict[init_method]
        for key, value in cur_dict.items():
            cur_dict[key] = np.array(value)
            cur_final[key] = np.mean(cur_dict[key])
            if key in ['add', 'adds']:
                cur_final[f'{key}_auc'] = bs.cal_auc(cur_dict[key])

    for init_method in ['global', 'rotate_60']:
        print('Init:', init_method)
        for key, value in final_dict[init_method].items():
            print(f'{key}: {value}')


if __name__ == "__main__":
    main()
