import numpy as np
import pickle
import h5py
import os
import argparse

import _init_paths
from scipy.spatial.transform import Rotation as srot
from scipy.optimize import least_squares
from common.aligning import estimateSimilarityTransform, estimateSimilarityUmeyama
from common.d3_utils import rotate_pts, scale_pts, transform_pts, rot_diff_rad, rot_diff_degree, rotate_about_axis, rotate_points_with_rotvec, compose_rt, align_rotation, transform_pcloud, transform_points
from common.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list, plot_hand_w_object
from common.debugger import *

from global_info import global_info
infos     = global_info()
sym_type  = infos.sym_type

def compute_pose_ransac(nocs_gt, nocs_pred, pcloud, partidx, num_parts, basename, r_raw_err, t_raw_err, s_raw_err, scale_gt=None, rt_gt=None, partidx_gt=None,
            align_sym=False, target_category=None, is_special=False, verbose=False):
    source_gt  = nocs_gt
    scale_dict = {'gt': [], 'baseline': [], 'nonlinear': []}
    r_dict     = {'gt': [], 'baseline': [], 'nonlinear': []}
    t_dict     = {'gt': [], 'baseline': [], 'nonlinear': []}
    xyz_err    = {'baseline': [], 'nonlinear': []}
    rpy_err    = {'baseline': [], 'nonlinear': []}
    scale_err  = {'baseline': [], 'nonlinear': []}

    if rt_gt is None:
        rt_gt = [None]
        scale_gt = [[1]]
        for j in range(1, num_parts):
            s, r, t, rt = estimateSimilarityUmeyama(nocs_gt[partidx_gt[j], :].transpose(), pcloud[partidx_gt[j], :].transpose())
            rt_gt.append(compose_rt(r, t))
            scale_gt.append([s])
    # if in cellphone or remote, find the closest NOCS prediction, not is_special means it has limited symmetry
    if not is_special:
        candidates = []
        mean_errs  = []
        for key, M in sym_type[target_category].items():
            for k in range(M):
                rmat = rotate_about_axis(2 * np.pi * k / M, axis=key)
                nocs_aligned = np.matmul(rmat, nocs_pred[:, 3:6].T-0.5) + 0.5
                candidates.append(nocs_aligned.T)
                mean_errs.append(np.linalg.norm(candidates[-1][partidx_gt[j], :] - nocs_gt[partidx_gt[1], :], axis=1).mean())
        best_ind = np.argmin(np.array(mean_errs))
        print(f'---best ind is {best_ind}')
        nocs_pred[:, 3:6] = candidates[best_ind]

    # single model estimation
    for j in range(1, num_parts):

        source0 = nocs_pred[partidx[j], 3*j:3*(j+1)]
        target0 = pcloud[partidx[j], :]
        print('source has shape: ', source0.shape)

        niter = 300
        inlier_th = 0.1

        dataset = dict()
        dataset['source'] = source0
        dataset['target'] = target0
        dataset['nsource'] = source0.shape[0]
        best_model, best_inliers = ransac(dataset, single_transformation_estimator, single_transformation_verifier, inlier_th, niter)
        tv1, tv2, rdiff = rot_diff_degree(best_model['rotation'], rt_gt[j][:3, :3], up=is_special)
        tdiff = np.linalg.norm(best_model['translation']-rt_gt[j][:3, 3])
        sdiff = np.linalg.norm(best_model['scale']-scale_gt[j][0])

        gt_canon      = np.concatenate([nocs_gt[partidx_gt[j], :], np.array([[0.5, 0.5, 0.5]])], axis=0)
        gt_pcloud     = transform_pcloud(gt_canon, compose_rt(rt_gt[j][:3, :3].T, rt_gt[j][:3, 3], s=scale_gt[j][0]))
        fitted_pcloud = transform_pcloud(np.concatenate([nocs_gt[partidx_gt[j], :], np.array([[0.5, 0.5, 0.5]])], axis=0), compose_rt(best_model['rotation'].T, best_model['translation'], s=best_model['scale']))
        tdiff = np.linalg.norm(gt_pcloud[-1] - fitted_pcloud[-1])
        if verbose:
            print('---visualizing pose predictions')
            gt_vect= {'p': gt_pcloud[-1], 'v': tv2}
            fitted_vect = {'p': fitted_pcloud[-1], 'v': tv1}
            plot3d_pts([[gt_pcloud, fitted_pcloud]], [['GT', 'pred']], s=3**2, arrows=[[fitted_vect, gt_vect]], title_name=['Check Pose'], limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 1.5]])
        print('part %d -- rdiff: %f degree, tdiff: %f, sdiff %f, ninliers: %f, npoint: %f' % (j, rdiff, tdiff, sdiff, np.sum(best_inliers), best_inliers.shape[0]))
        target0_fit = best_model['scale'] * np.matmul(best_model['rotation'], source0.T) + best_model['translation'].reshape((3, 1))
        target0_fit = target0_fit.T
        best_model0 = best_model
        rpy_err['baseline'].append(rdiff)
        xyz_err['baseline'].append(tdiff)
        scale_err['baseline'].append(sdiff)
        r_raw_err['baseline'][j].append(rdiff)
        t_raw_err['baseline'][j].append(tdiff)
        s_raw_err['baseline'][j].append(sdiff)

        scale_dict['baseline'].append(best_model0['scale'])
        r_dict['baseline'].append(best_model0['rotation'])
        t_dict['baseline'].append(best_model0['translation'])
        scale_dict['gt'].append(scale_gt[j][0])
        r_dict['gt'].append(rt_gt[j][:3, :3])
        t_dict['gt'].append(rt_gt[j][:3, 3])

    rts_dict = {}
    rts_dict['scale']   = scale_dict
    rts_dict['rotation']      = r_dict
    rts_dict['translation']   = t_dict
    rts_dict['xyz_err'] = xyz_err
    rts_dict['rpy_err'] = rpy_err
    rts_dict['scale_err'] = scale_err

    return rts_dict
    # except:
    #     print(f'wrong entry with {i}th data!!')

def ransac(dataset, model_estimator, model_verifier, inlier_th, niter=10000):
    best_model = None
    best_score = -np.inf
    best_inliers = None
    for i in range(niter):
        cur_model = model_estimator(dataset)
        cur_score, cur_inliers = model_verifier(dataset, cur_model, inlier_th)
        if cur_score > best_score:
            best_model = cur_model
            best_inliers = cur_inliers
    best_model = model_estimator(dataset, best_inliers)
    return best_model, best_inliers

def single_transformation_estimator(dataset, best_inliers = None):
    # dataset: dict, fields include source, target, nsource
    if best_inliers is None:
        sample_idx = np.random.randint(dataset['nsource'], size=3)
    else:
        sample_idx = best_inliers
    rotation, scale, translation = transform_pts(dataset['source'][sample_idx,:], dataset['target'][sample_idx,:])
    strans = dict()
    strans['rotation'] = rotation
    strans['scale'] = scale
    strans['translation'] = translation
    return strans

def single_transformation_verifier(dataset, model, inlier_th):
    # dataset: dict, fields include source, target, nsource, ntarget
    # model: dict, fields include rotation, scale, translation
    res = dataset['target'].T - model['scale'] * np.matmul( model['rotation'], dataset['source'].T ) - model['translation'].reshape((3, 1))
    inliers = np.sqrt(np.sum(res**2, 0)) < inlier_th
    score = np.sum(inliers)
    return score, inliers

def objective_eval(params, x0, y0, x1, y1, joints, isweight=True):
    # params: [:3] R0, [3:] R1
    # x0: N x 3, y0: N x 3, x1: M x 3, y1: M x 3, R0: 1 x 3, R1: 1 x 3, joints: K x 3
    rotvec0 = params[:3].reshape((1,3))
    rotvec1 = params[3:].reshape((1,3))
    res0 = y0 - rotate_points_with_rotvec(x0, rotvec0)
    res1 = y1 - rotate_points_with_rotvec(x1, rotvec1)
    res_joint = rotate_points_with_rotvec(joints, rotvec0) - rotate_points_with_rotvec(joints, rotvec1)
    if isweight:
        res0 /= x0.shape[0]
        res1 /= x1.shape[0]
        res_joint /= joints.shape[0]
    return np.concatenate((res0, res1, res_joint), 0).ravel()

def joint_transformation_estimator(dataset, best_inliers = None):
    # dataset: dict, fields include source0, target0, nsource0,
    #     source1, target1, nsource1, joint_direction
    if best_inliers is None:
        sample_idx0 = np.random.randint(dataset['nsource0'], size=3)
        sample_idx1 = np.random.randint(dataset['nsource1'], size=3)
    else:
        sample_idx0 = best_inliers[0]
        sample_idx1 = best_inliers[1]

    source0 = dataset['source0'][sample_idx0, :]
    target0 = dataset['target0'][sample_idx0, :]
    source1 = dataset['source1'][sample_idx1, :]
    target1 = dataset['target1'][sample_idx1, :]

    # prescaling and centering
    scale0 = scale_pts(source0, target0)
    scale1 = scale_pts(source1, target1)
    scale0_inv = scale_pts(target0, source0) # check if could simply take reciprocal
    scale1_inv = scale_pts(target1, source1)

    target0_scaled_centered = scale0_inv*target0
    target0_scaled_centered -= np.mean(target0_scaled_centered, 0, keepdims=True)
    source0_centered = source0 - np.mean(source0, 0, keepdims=True)

    target1_scaled_centered = scale1_inv*target1
    target1_scaled_centered -= np.mean(target1_scaled_centered, 0, keepdims=True)
    source1_centered = source1 - np.mean(source1, 0, keepdims=True)

    joint_points0 = np.ones_like(np.linspace(0, 1, num = np.min((source0.shape[0], source1.shape[0]))+1 )[1:].reshape((-1, 1)))*dataset['joint_direction'].reshape((1, 3))
    joint_points1 = np.ones_like(np.linspace(0, 1, num = np.min((source0.shape[0], source1.shape[0]))+1 )[1:].reshape((-1, 1)))*dataset['joint_direction'].reshape((1, 3))

    R0 = rotate_pts(source0_centered, target0_scaled_centered)
    R1 = rotate_pts(source1_centered, target1_scaled_centered)
    rdiff0 = np.inf
    rdiff1 = np.inf
    niter = 10
    degree_th = 0.05
    isalternate = False

    if not isalternate:
        rotvec0 = srot.from_dcm(R0).as_rotvec()
        rotvec1 = srot.from_dcm(R1).as_rotvec()
        res = least_squares(objective_eval, np.hstack((rotvec0, rotvec1)), verbose=0, ftol=1e-4, method='lm',
                        args=(source0_centered, target0_scaled_centered, source1_centered, target1_scaled_centered, joint_points0, False))
        R0 = srot.from_rotvec(res.x[:3]).as_dcm()
        R1 = srot.from_rotvec(res.x[3:]).as_dcm()
    else:
        for i in xrange(niter):
            if rdiff0<=degree_th and rdiff1<=degree_th:
                break
            newsrc0 = np.concatenate( (source0_centered, joint_points0), 0 )
            newtgt0 = np.concatenate( (target0_scaled_centered, np.matmul( joint_points0, R1.T ) ), 0 )
            newR0 = rotate_pts( newsrc0, newtgt0 )
            rdiff0 = rot_diff_degree(R0, newR0)
            R0 = newR0

            newsrc1 = np.concatenate( (source1_centered, joint_points1), 0 )
            newtgt1 = np.concatenate( (target1_scaled_centered, np.matmul( joint_points1, R0.T ) ), 0 )
            newR1 = rotate_pts( newsrc1, newtgt1 )
            rdiff1 = rot_diff_degree(R1, newR1)
            R1 = newR1

    translation0 = np.mean(target0.T-scale0*np.matmul(R0, source0.T), 1)
    translation1 = np.mean(target1.T-scale1*np.matmul(R1, source1.T), 1)
    jtrans = dict()
    jtrans['rotation0'] = R0
    jtrans['scale0'] = scale0
    jtrans['translation0'] = translation0
    jtrans['rotation1'] = R1
    jtrans['scale1'] = scale1
    jtrans['translation1'] = translation1
    return jtrans

def joint_transformation_verifier(dataset, model, inlier_th):
    # dataset: dict, fields include source, target, nsource, ntarget
    # model: dict, fields include rotation, scale, translation
    res0 = dataset['target0'].T - model['scale0'] * np.matmul( model['rotation0'], dataset['source0'].T ) - model['translation0'].reshape((3, 1))
    inliers0 = np.sqrt(np.sum(res0**2, 0)) < inlier_th
    res1 = dataset['target1'].T - model['scale1'] * np.matmul( model['rotation1'], dataset['source1'].T ) - model['translation1'].reshape((3, 1))
    inliers1 = np.sqrt(np.sum(res1**2, 0)) < inlier_th
    score = ( np.sum(inliers0)/res0.shape[0] + np.sum(inliers1)/res1.shape[0] ) / 2
    return score, [inliers0, inliers1]
