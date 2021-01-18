import numpy as np
import json
import h5py
import pickle
import argparse
import platform
import torch.nn.functional as F
from sklearn.decomposition import PCA
from manopth import rodrigues_layer
from manopth.manolayer import ManoLayer
try:
    from PIL import Image
except ImportError:
    print('Could not import PIL in handutils')

import _init_paths
from common.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list, plot_hand_w_object

def get_annot_scale(annots, visibility=None, scale_factor=2.2):
    """
    Retreives the size of the square we want to crop by taking the
    maximum of vertical and horizontal span of the hand and multiplying
    it by the scale_factor to add some padding around the hand
    """
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    s = max_delta * scale_factor
    return s

def get_annot_center(annots, visibility=None):
    # Get scale
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    return np.asarray([c_x, c_y])


def transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts(np.ndarray): (point_nb, 2)
    """
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows.astype(int)


def transform_img(img, affine_trans, res):
    """
    Args:
    center (tuple): crop center coordinates
    scale (int): size in pixels of the final crop
    res (tuple): final image size
    """
    trans = np.linalg.inv(affine_trans)

    img = img.transform(
        tuple(res), Image.AFFINE, (trans[0, 0], trans[0, 1], trans[0, 2],
                                   trans[1, 0], trans[1, 1], trans[1, 2]))
    return img


def get_affine_transform(center, scale, res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [
        1,
    ])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -res[1] / 2
    t_mat[1, 2] = -res[0] / 2
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [
        1,
    ])
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2],
                                                   scale, res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(
        np.float32)


def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[1]) / scale
    affinet[1, 1] = float(res[0]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    affinet[2, 2] = 1
    return affinet


def get_affine_transform_bak(center, scale, res, rot):
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / scale
    t[1, 1] = float(res[0]) / scale
    t[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    t[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    t[2, 2] = 1
    if rot != 0:
        rot_mat = np.zeros((3, 3))
        sn, cs = np.sin(rot), np.cos(rot)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t))).astype(np.float32)
    return t, t


def get_hand_mesh_canonical(hf, verbose=False):
    """
    verts, faces, joints:
        1. params + mano model;
        2. prediction/GT directly'
    """
    hand_joints_gt   = hf['handjoints_gt'][()].reshape(-1, 3)
    hand_joints_pred = hf['handjoints_pred'][()].reshape(-1, 3)
    hand_verts_gt    = hf['handvertices_gt'][()].reshape(-1, 3)
    hand_verts_pred  = hf['handvertices_pred'][()].reshape(-1, 3)
    hand_params_gt   = hf['regression_params_gt'][()].reshape(1, -1)
    hand_params_pred = hf['regression_params_pred'][()].reshape(1, -1)
    if hand_params_pred.shape[0] < 51:
        hand_rotation = hf['regressionR_pred'][()]
        rot_in_6d   = hand_rotation[:, :2].T.reshape(1, -1)
        hand_params_pred  = np.asarray(np.concatenate([rot_in_6d, hand_params_pred.reshape(1, -1)], axis=1)).reshape(1, -1)#

    scale = 200
    verts = []
    faces = []
    joints= []

    if verbose:
        mano_layer_right = ManoLayer(
                mano_root=mano_path , side='right', root_rot_mode='ratation_6d', use_pca=False, ncomps=45, flat_hand_mean=True)
        for hand_params in [hand_params_gt, hand_params_pred]:
            hand_vertices, hand_joints = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(hand_params))
            hand_joints = hand_joints.cpu().data.numpy()[0]/scale
            hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale
            hand_faces = mano_layer_right.th_faces.cpu().data.numpy()
            verts.append(hand_vertices)
            faces.append(hand_faces)
            joints.append(hand_joints)
    else:
        verts = [hand_verts_gt, hand_verts_pred]
        joints= [hand_joints_gt, hand_joints_pred]
    if verbose:
        # compute GT joint error, and GT vertices error
        plot_hand_w_object(obj_verts=verts[0], obj_faces=faces[0], hand_verts=verts[1], hand_faces=faces[1], s=5, save=False)

    return verts, faces, joints

def get_hand_mesh_camera(verts_canon, faces_canon, joints_canon, hand_trans, verbose=False):
    """
    we only predict rotation, scale, and
    so we further need a T to get camera setting
    """
    verts, faces, joints = [], [], []
    for j in range(2):
        verts.append(verts_canon[j] + hand_trans[j])
        joints.append(joints_canon[j] + hand_trans[j])
    faces = faces_canon
    if verbose:
        plot_hand_w_object(obj_verts=verts[0], obj_faces=faces[0], hand_verts=verts[1], hand_faces=faces[1], s=5, mode='continuous', save=False)

    return verts, faces, joints

def vote_hand_joints_cam(hf, input_pts, part_idx_list_gt, verbose=False):
    """
    heat + vec --> joints
    """
    hand_heatmap_gt     = hf['handheatmap_per_point_gt'][()].transpose(1, 0)
    hand_unitvec_gt     = hf['handunitvec_per_point_gt'][()].transpose(1, 0) # todo
    hand_heatmap_pred   = hf['handheatmap_per_point_pred'][()].transpose(1, 0)
    hand_unitvec_pred   = hf['handunitvec_per_point_pred'][()].transpose(1, 0)
    thres_r             = 0.2
    thres_var           = 0.05
    results = {'vote': [], 'divergence': [], 'regression':None, 'final': []}
    for j in range(21):
        offset_gt    = hand_unitvec_gt[:, 3*j:3*(j+1)]   * (1- hand_heatmap_gt[:, j:j+1].reshape(-1, 1)) * thres_r
        offset_pred  = hand_unitvec_pred[:, 3*j:3*(j+1)] * (1- hand_heatmap_pred[:, j:j+1].reshape(-1, 1)) * thres_r

        idx = part_idx_list_gt[3]
        ind_vote = np.where(hand_heatmap_gt[idx, j:j+1] > 0.01)[0]
        if len(ind_vote) ==0:
            results['vote'].append(np.array([0, 0, 0]).reshape(1, 3))
            results['divergence'].append(1)
            continue
        joint_votes      = input_pts[idx][ind_vote] + offset_pred[idx][ind_vote]
        vote_var         = np.sum(np.var(joint_votes, axis=0)) # N, 3
        joint_voted      = np.mean(joint_votes, axis=0).reshape(1, 3)
        results['vote'].append(joint_voted)
        results['divergence'].append(vote_var)
        print(f'Joint {j} GT: \n', offset_gt[idx][ind_vote])
        print('')
        print(f'Joint {j} pred: \n', offset_pred[idx][ind_vote])
        if verbose:
            plot_arrows(input_pts[idx][ind_vote], [offset_gt[idx][ind_vote], offset_pred[idx][ind_vote]], whole_pts=input_pts, title_name='{}th_joint_voting'.format(j), limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], dpi=200, s=25, thres_r=0.2, sparse=True, save=args.save_fig, index=i)

    return results

def get_hand_regression_camera(hf, input_pts, verbose=False):
    """
    get_hand_regression_camera: 3d joints
    """
    hand_joints_gt   = hf['regression_params_gt'][()].reshape(-1, 3)
    hand_joints_pred = hf['regression_params_pred'][()].reshape(-1, 3)
    print('')
    joints = [hand_joints_gt, hand_joints_pred]
    if verbose:
        print('hand_joints_gt: \n', hand_joints_gt)
        print('hand_joints_pred: \n', hand_joints_pred)
        print('---hand_joints regression error: \n', np.linalg.norm(hand_joints_pred-hand_joints_gt, axis=1))
        plot3d_pts([[input_pts, hand_joints_gt, hand_joints_pred]], [['Input pts', 'GT joints', 'Pred joints']], s=8**2, title_name=['Direct joint regression'], limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
#
def get_hand_trans2camera(hf, basename, joints_canon, extra_hf=None, category='eyeglasses', verbose=False):
    """
    get_hand_trans2camera: T
    """
    if extra_hf is not None:
        hand_trans_gt = extra_hf['regressionT_gt'][()] + extra_hf['mean_hand_gt'][()]
        hand_trans_pred = extra_hf['regressionT_pred'][()] + extra_hf['mean_hand_gt'][()]
    else:
        if 'hand_joints_gt' in list(hf.keys()):
            hand_joints   = hf['hand_joints_gt'][()]
            hand_contacts = hf['hand_contacts_gt'][()]
        else:
            h5_file = f'{second_path}/data/hdf5/{category}/{basename[:-2]}/{basename[-1]}.h5'
            with h5py.File(h5_file, 'r') as handle:
                hand_joints   = handle['joints'][()]
                hand_contacts = handle['contacts'][()]
        hand_trans_gt = np.mean(hand_joints - joints_canon[0], axis=0).reshape(1, 3) # use GT joints
        hand_trans_pred = hand_trans_gt

    return [hand_trans_gt, hand_trans_pred]


def get_contacts_camera_raw(hf, basename=None, category='eyeglasses', verbose=False):
    #>>>>>>>>>>>>>>>>>>>> get gt
    if 'hand_contacts_gt' in list(hf.keys()):
        hand_contacts = hf['hand_contacts_gt'][()]
    else:
        h5_file = f'{second_path}/data/hdf5/{category}/{basename[:-2]}/{basename[-1]}.h5'
        with h5py.File(h5_file, 'r') as handle:
            hand_contacts = handle['contacts'][()]

    return [hand_contacts, hand_contacts]

def get_contacts_camera(hf, basename, verbose=False):
    # box_label_mask_gt (64,)
    # center_label_gt (64, 3)
    # center_pred: 128, 3
    # object_assignment_pred (128,)
    # objectness_label_pred (128,)
    # objectness_mask_pred (128,)
    # objectness_scores_pred (128, 2)
    coords = hf['center_label_gt'][()]
    mask   = hf['box_label_mask_gt'][()]
    coords_gt = coords[np.where(mask>0)[0], :]

    coords =  hf['center_pred'][()]
    mask   =  hf['objectness_label_pred'][()]
    confidence = hf['center_confidence_pred'][()]
    scores = F.softmax(torch.FloatTensor(hf['objectness_scores_pred'][()]), dim=1).cpu().numpy()

    # >>>>>>>>>>>>> check confidence distribution
    valid_inds = np.where(scores[:, 1]>0.8)[0]
    # valid_ind = scores[:, 1].argsort()[-10:][::-1]
    # valid_ind   = confidence[valid_inds].argsort()[-5:][::-1]
    coords_pred = coords[valid_inds, :]

    gt_color = np.ones_like(coords_gt)
    gt_color[:, 0:2] = 0 # Blue
    pred_color = np.ones_like(coords_pred)
    pred_color[:, 1:3] = 0 # Red
    objectness_color = np.copy(pred_color) * scores[valid_inds, 1].reshape(-1, 1)
    confidence_color = np.copy(pred_color) * confidence[valid_inds].reshape(-1, 1)

    # >>>>>>>>>>>>>>>> NMS
    coords_final = np.copy(coords_gt)
    coords_idx  = []
    for j in range(coords_gt.shape[0]):
        idx = np.where(np.linalg.norm(coords_pred - coords_gt[j:j+1, :], axis=1)<0.1)[0]
        if len(idx) > 0:
            choose_id       = confidence[valid_inds][idx].argsort()[-1]
            coords_final[j] = coords_pred[idx, :][choose_id]
            coords_idx.append(idx[choose_id])

    if verbose:
        print(confidence[valid_inds])
        print(confidence[valid_inds][coords_idx[:]])
        confidence_color_final = confidence_color[coords_idx[:]]
        coords_pred_final = coords_pred[coords_idx[:], :]
        plot3d_pts([[coords_gt], [coords_gt, coords_pred]], [['GT'], ['GT', 'Pred']], s=[10**2, 5, 5], mode='continuous', dpi=200, title_name=['contact pts', 'Objectness'], color_channel=[[gt_color], [gt_color, objectness_color]])
        plot3d_pts([[coords_gt], [coords_gt, coords_pred]], [['GT'], ['GT', 'Pred']], s=[10**2, 5, 5], mode='continuous', dpi=200, title_name=['contact pts', 'Confidence'], color_channel=[[gt_color], [gt_color, confidence_color]])
        plot3d_pts([[coords_gt], [coords_gt, coords_pred_final]], [['GT'], ['GT', 'Pred']], s=[10**2, 5, 5], mode='continuous', dpi=200, title_name=['contact pts', 'Final'], color_channel=[[gt_color], [gt_color, confidence_color_final]])

    return [coords_gt, coords_final], [coords_gt.shape[0], len(coords_idx)]

def get_contacts_vote(hf, basename, input_pts, verbose=False):
    # vote_label_gt (2048, 9)
    # vote_label_mask_gt (2048,)
    # vote_xyz_pred (1024, 3)
    # seed_xyz_pred (1024, 3)
    # seed_inds_pred (1024)
    offset_gt = hf['vote_label_gt'][()][:, :3]
    mask      = hf['vote_label_mask_gt'][()]
    idx       = np.where(mask>0)[0]
    ind_vote  = hf['seed_inds_pred'][()]
    offset_pred =  hf['vote_xyz_pred'][()][:, :3] - hf['seed_xyz_pred'][()][:, :3]
    if verbose:
        plot_arrows(input_pts[ind_vote], [offset_gt[ind_vote], offset_pred], whole_pts=input_pts, title_name='contact voting', dpi=200, s=1, thres_r=0.2, sparse=True)
