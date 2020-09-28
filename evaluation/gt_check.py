"""
Log:
Instructions:
#>>>>>>>>>>>>> <<<<<<<<<<<<<<< #
cd dataset && python base.py
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/eyeglasses/1/viz/* ~/Documents/ICML2021/model/eyeglasses/1/viz
cd evaluation && python gt_check.py
#>>>>>>>>>>>>>> <<<<<<<<<<<<<< #

9.26:
    - add contacts & joints check;
    - 

"""

import numpy as np
import os
import sys
import time
import json
import h5py
import pickle
import argparse
import platform

import platform
from sklearn.decomposition import PCA
from manopth import rodrigues_layer
from manopth.manolayer import ManoLayer
import torch
import matplotlib.pyplot as plt
from pytransform3d.rotations import *

import _init_paths
from common.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list, plot_hand_w_object
from common.data_utils import fast_load_obj, get_obj_mesh
from common.d3_utils import mat_from_rvec, rvec_from_mat, compute_rotation_matrix_from_euler, compute_euler_angles_from_rotation_matrices, compute_rotation_matrix_from_ortho6d
from common.debugger import breakpoint, print_group
from global_info import global_info

infos     = global_info()
base_path   = infos.base_path
second_path = infos.second_path
render_path = infos.render_path
viz_path  = infos.viz_path
hand_mesh = infos.hand_mesh
hand_urdf = infos.hand_urdf
grasps_meta  = infos.grasps_meta
mano_path    = infos.mano_path

whole_obj = infos.whole_obj
part_obj  = infos.part_obj
obj_urdf  = infos.obj_urdf


def transform_pcloud(pcloud, RT, inv=False, extra_R=None, verbose=False):
    """
    by default, pcloud: [N, 3]
    """
    # if extra_R is not None:
    #     pcloud = np.dot(pcloud, extra_R[:3, :3].T)
    if inv:
        inv_R     = np.linalg.pinv(RT[:3, :3])
        pcloud_tf = np.dot(inv_R, pcloud.T  - RT[:3, 3].reshape(3, 1))
        pcloud_tf = pcloud_tf.T
    else:
        pcloud_tf = np.dot(pcloud, RT[:3, :3].T) + RT[:3, 3].reshape(1, 3)
    if verbose:
        print_group([RT, pcloud[:3, :], pcloud_tf[:3, :]], ['RT', 'original pts', 'transformed pts'])
        plot3d_pts([[pcloud, pcloud_tf]], [['pts', 'transformed pts']], s=1, mode='continuous',  limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 1.5]], title_name=['Camera + World Pts'])

    return pcloud_tf

def get_hand_pcloud(data, verbose=False):
    roi_ind = np.where(data['partcls_per_point']==3)[0]
    hand_pcloud = data['P'][roi_ind]
    if verbose:
        plot3d_pts([[hand_pcloud]], [['hand pcloud']], s=1, mode='continuous', limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 1.5]])

    return hand_pcloud

def get_obj_pcloud(data, category='eyeglasses', verbose=False):
    """"""
    if category == 'eyeglasses':
        hand_ind = 3
    roi_ind = np.where(data['partcls_per_point']<hand_ind)[0]
    obj_pcloud = data['P'][roi_ind]
    obj_pcloud_list = []
    nocs_list = []
    for j in range(hand_ind):
        roi_ind = np.where(data['partcls_per_point']==j)[0]
        obj_pcloud_list.append(data['P'][roi_ind])
        if j <hand_ind:
            nocs_list.append(data['nocs_per_point'][roi_ind])
    if verbose:
        plot3d_pts([[obj_pcloud]], [['object_pcloud']], s=2**2, mode='continuous',  title_name=['pcloud camera'])
        plot3d_pts([obj_pcloud_list], [['obj_part0', 'obj_part1', 'obj_part2', 'hand']], s=2, mode='continuous', title_name=['pcloud segmentation'])
        plot3d_pts([[nocs_list[0]], [nocs_list[1]], [nocs_list[2]]], [['nocs0'], ['nocs1'], ['nocs1']], color_channel=[[nocs_list[0]], [nocs_list[1]], [nocs_list[2]]], s=2, mode='continuous', limits = [[0, 1], [0, 1], [0, 1]], axis_off=True, title_name=['NOCS']*3)

    return obj_pcloud, obj_pcloud_list, nocs_list


# def get_hand_mesh_onestep(data, mano_layer, verbose=False):
#     verts = []
#     faces = []
#     hand_trans = data['regressionT'].reshape(1, -1)
#     # hand_trans = np.array([0, 0, 0]).reshape(1, -1)
#     hand_poses = data['regression_params'].reshape(1, -1)
#     print_group([hand_poses, hand_trans], ['mano_poses', 'mano_trans'])
#     scale = 200
#     #>>>>>>>>>>> core code
#     rot2blender = matrix_from_euler_xyz([-np.pi/2, 0, 0])
#     rot_canon   = mat_from_rvec(hand_poses[0, :3])
#     print(data.keys())
#     rot2camera  = data['extrinsic_params'][:3, :3]
#     trans2camera= data['extrinsic_params'][:3, 3]
#     hand_poses[0, :3] = rvec_from_mat(rot2camera @ rot2blender @ rot_canon) #
#     hand_trans  = rot2camera @ rot2blender @ hand_trans.reshape(3, 1) * 1000 / scale + trans2camera.reshape(3, 1)
#     hand_trans  = (hand_trans / (1000/scale)).reshape(1, 3)
#
#     hand_vertices, hand_joints = mano_layer.forward(th_pose_coeffs=torch.FloatTensor(hand_poses), th_trans=torch.FloatTensor(hand_trans))
#     hand_joints = hand_joints.cpu().data.numpy()[0]/scale
#     hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale
#
#     print(f'hand vertices have {hand_vertices.shape[0]} pts')
#     hand_faces = mano_layer.th_faces.cpu().data.numpy()
#     print(f'hand faces {hand_faces.shape} is {hand_faces[:3, :]}')
#     verts.append(hand_vertices)
#     faces.append(hand_faces)
#
#     if verbose:
#         plot_hand_w_object(obj_verts=verts[0], obj_faces=faces[0], hand_verts=verts[0], hand_faces=faces[0],save=False, mode='continuous')
#
#     return verts[0], faces[0]

def get_hand_mesh(data, mano_layer=None, verbose=False):
    verts = []
    faces = []
    hand_trans = data['regressionT'].reshape(1, -1)
    hand_poses = data['regression_params'].reshape(1, -1)

    #>>>>>>>>>>> core code
    alpha = -np.pi/2
    rot2blender = matrix_from_euler_xyz([alpha, 0, 0])
    identity_mat= matrix_from_euler_xyz([0, 0, 0])
    rot_canon   = mat_from_rvec(hand_poses[0, :3])
    rot2camera  = data['extrinsic_params'][:3, :3]
    trans2camera= data['extrinsic_params'][:3, 3]
    # # test 6d rotation transform
    # recovered_mat = compute_rotation_matrix_from_ortho6d(torch.FloatTensor(rot_canon[:, :2].T.reshape(1, -1)).cuda())
    # print_group([rot_canon, recovered_mat[0]], ['original mat', 'recovered mat'])

    # # test when T=0, or R=0
    # hand_trans = np.zeros((1, 3))
    # hand_poses[0, :3] = rvec_from_mat(identity_mat) # supervise
    print_group([hand_poses, hand_trans], ['mano_poses', 'mano_trans'])
    scale = 200

    #>>>>>>>>>>>>>>>>> working example
    # hand_poses[0, :3] = rvec_from_mat(rot2blender @ rot_canon) #
    # hand_trans  = (rot2blender @ hand_trans.reshape(3, 1)).reshape(1, -1)
    hand_vertices, hand_joints = mano_layer.forward(th_pose_coeffs=torch.FloatTensor(hand_poses), th_trans=torch.FloatTensor(hand_trans))
    hand_joints = hand_joints.cpu().data.numpy()[0]/scale
    hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale
    hand_vertices = np.dot(rot2camera @ rot2blender, hand_vertices.T).T + trans2camera.reshape(1, 3) # transform by RT in camera space
    hand_faces = mano_layer.th_faces.cpu().data.numpy()
    print(f'hand faces {hand_faces.shape} is {hand_faces[:3, :]}')
    verts.append(hand_vertices)
    faces.append(hand_faces)

    # #>>>>>>>>>>>>>>>>> working example but lack translation
    # hand_poses[0, :3] = rvec_from_mat(rot2camera @ rot2blender @ rot_canon) #
    # hand_vertices, hand_joints = mano_layer.forward(th_pose_coeffs=torch.FloatTensor(hand_poses), th_trans=torch.FloatTensor(np.zeros((1, 3))))
    # hand_joints = hand_joints.cpu().data.numpy()[0]/scale
    # hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale
    # hand_vertices = hand_vertices #+ trans2camera.reshape(1, 3) # transform by RT in camera space
    #

    mano_layer_rot6d = ManoLayer(
            mano_root=mano_path , root_rot_mode='ratation_6d', side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
    rot_in_6d  = (rot2camera @ rot2blender @ rot_canon)[:, :2].T.reshape(1, -1)
    hand_poses = np.concatenate([rot_in_6d, hand_poses[:, 3:]], axis=1)
    # hand_poses[0, :3] = rvec_from_mat(rot2blender @ rot_canon) #
    # hand_trans  = (rot2blender @ hand_trans.reshape(3, 1)).reshape(1, -1)
    # hand_trans = np.array([0, 1, 1*200/1000]).reshape(1, -1)
    print('hand_trans before mano: ', hand_trans)
    hand_vertices, hand_joints = mano_layer_rot6d.forward(th_pose_coeffs=torch.FloatTensor(hand_poses), th_trans=torch.FloatTensor(hand_trans))
    hand_joints = hand_joints.cpu().data.numpy()[0]/scale
    hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale

    verts.append(hand_vertices)
    faces.append(hand_faces)
    print_group([verts[1]- verts[0], np.eye(3) - rot2blender], ['vertices error', 'rot error'])
    print_group([(np.linalg.pinv(np.eye(3) - rot2blender) @ (verts[1]- verts[0]).T).T], ['T0'])
    print(f'hand vertices have {hand_vertices.shape[0]} pts')
    if verbose:
        plot_hand_w_object(obj_verts=verts[0], obj_faces=faces[0], hand_verts=verts[1], hand_faces=faces[1],save=False, mode='continuous')

    return verts[0], faces[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--exp_num', default='1', required=False)
    args = parser.parse_args()
    #
    my_dir = second_path + f'/model/{args.item}/{args.exp_num}/viz/'
    # /home/dragon/Documents/ICML2021/model/eyeglasses/1/viz

    viz_files = os.listdir(my_dir)
    mano_layer_right = ManoLayer(
            mano_root=mano_path , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    for i in range(len(viz_files)):
        viz_file = my_dir + viz_files[i]
        data = np.load(viz_file, allow_pickle=True).item()
        # >>>>>>>>> we get from rendered data
        hand_pcloud = get_hand_pcloud(data, verbose=True)
        obj_pcloud, obj_pcloud_list, nocs_list  = get_obj_pcloud(data, verbose=True)
        hand_vertices, hand_faces = get_hand_mesh(data, mano_layer_right, verbose=True)
        obj_vertices, obj_faces   = get_obj_mesh(viz_files[i].split('.')[0], verbose=False)
        obj_vertices_tf           = transform_pcloud(np.copy(obj_vertices), RT=data['extrinsic_params'], extra_R= matrix_from_euler_xyz([-np.pi/2, 0, 0]))

        hand_contacts = data['hand_contacts'][()]
        hand_pcloud_tf = transform_pcloud(np.copy(hand_pcloud), RT=data['extrinsic_params'], inv=True,  verbose=False)
        obj_pcloud_tf  = transform_pcloud(np.copy(obj_pcloud), RT=data['extrinsic_params'], inv=True,  verbose=False)
        #
        plot_hand_w_object(obj_verts=hand_vertices, obj_faces=hand_faces, hand_verts=hand_vertices, hand_faces=hand_faces, s=5**2, pts=[[hand_contacts]], save=False, mode='continuous')

        plot_hand_w_object(obj_verts=obj_vertices, obj_faces=obj_faces, hand_verts=obj_vertices, hand_faces=obj_faces, pts=[[obj_pcloud_tf]], save=False, mode='continuous')

        plot_hand_w_object(obj_verts=obj_vertices_tf, obj_faces=obj_faces, hand_verts=hand_vertices, hand_faces=hand_faces, save=False, mode='continuous')

        # hand_vertices_tf, hand_faces_tf = get_hand_mesh_onestep(data, mano_layer_right, verbose=False)
        # plot_hand_w_object(obj_verts=hand_vertices_tf, obj_faces=hand_faces_tf, hand_verts=hand_vertices_tf, hand_faces=hand_faces_tf, pts=[[hand_pcloud]], save=False, mode='continuous')


        plt.show()
        plt.close()






    #
