# import numpy as np
import os
import sys
import time
import json
import h5py
import pickle
import numpy as np
import argparse
import platform
import torch.nn.functional as F
from sklearn.decomposition import PCA
from manopth import rodrigues_layer
from manopth.manolayer import ManoLayer
import matplotlib.pyplot as plt
from pytransform3d.rotations import *
import torch

import _init_paths
from common.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list, plot_hand_w_object
from common.data_utils import get_demo_h5, get_full_test, save_objmesh, fast_load_obj, get_obj_mesh
from common.aligning import estimateSimilarityTransform, estimateSimilarityUmeyama
from evaluation.gt_check import transform_pcloud
from global_info import global_info

infos     = global_info()
my_dir    = infos.base_path
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
def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation.transpose()
    aligned_RT[:3, 3]  = translation
    aligned_RT[3, 3]   = 1
    return aligned_RT

def get_parts_ind(hf, num_parts=4, gt_key='partcls_per_point_gt', pred_key='partcls_per_point_pred', verbose=False):
    mask_gt        =  hf[gt_key][()]
    mask_pred      =  hf[pred_key][()]
    part_idx_list_gt     = []
    part_idx_list_pred   = []

    if len(mask_pred.shape) > 1:
        mask_pred = mask_pred.transpose(1, 0)
        cls_per_pt_pred      = np.argmax(mask_pred, axis=1)
    else:
        cls_per_pt_pred = mask_pred

    for j in range(num_parts):
        part_idx_list_gt.append(np.where(mask_gt==j)[0])
        part_idx_list_pred.append(np.where(cls_per_pt_pred==j)[0])

    return part_idx_list_gt, part_idx_list_pred

def get_parts_pcloud(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4, verbose=False):
    input_pts = hf['P_gt'][()]
    if input_pts.shape[1] > 10:
        input_pts = input_pts.transpose(1, 0)
    gt_parts  = [input_pts[part_idx_list_gt[j], :3] for j in range(num_parts)]
    pred_parts= [input_pts[part_idx_list_gt[j], :3] for j in range(num_parts)]
    input_pts = input_pts[:, :3]
    if verbose:
        plot3d_pts([gt_parts], [['Part {}'.format(j) for j in range(num_parts)]], s=5**2, title_name=['GT seg'], sub_name=str(i),  axis_off=False, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([pred_parts], [['Part {}'.format(j) for j in range(num_parts)]], s=5**2, title_name=['Pred seg'], sub_name=str(i), axis_off=False, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)

    return input_pts, gt_parts, pred_parts

def get_hand_mesh_canonical(hf, verbose=False):
    mano_layer_right = ManoLayer(
            mano_root=mano_path , side='right', root_rot_mode='ratation_6d', use_pca=False, ncomps=45, flat_hand_mean=True)
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
    for hand_params in [hand_params_gt, hand_params_pred]:
        hand_vertices, hand_joints = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(hand_params))
        hand_joints = hand_joints.cpu().data.numpy()[0]/scale
        hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale
        print(f'hand vertices have {hand_vertices.shape[0]} pts')
        hand_faces = mano_layer_right.th_faces.cpu().data.numpy()
        print(f'hand faces {hand_faces.shape} is {hand_faces[:3, :]}')
        verts.append(hand_vertices)
        faces.append(hand_faces)
        joints.append(hand_joints)

    if verbose:
        # compute GT joint error, and GT vertices error
        plot_hand_w_object(obj_verts=verts[0], obj_faces=faces[0], hand_verts=verts[1], hand_faces=faces[1], s=5**2, save=False)

    return verts, faces, joints

def get_hand_mesh_camera(verts_canon, faces_canon, joints_canon, hand_trans, verbose=False):
    verts, faces, joints = [], [], []
    for j in range(2):
        verts.append(verts_canon[j] + hand_trans[j])
        joints.append(joints_canon[j] + hand_trans[j])
    faces = faces_canon
    if verbose:
        plot_hand_w_object(obj_verts=verts[0], obj_faces=faces[0], hand_verts=verts[1], hand_faces=faces[1], s=5**2, save=False)

    return verts, faces, joints

def vote_hand_joints_cam(hf, input_pts, part_idx_list_gt, verbose=False):
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
def get_hand_trans2camera(hf, basename, joints_canon, category='eyeglasses', verbose=False):
    #>>>>>>>>>>>>>>>>>>>> get gt
    if 'hand_joints_gt' in list(hf.keys()):
        hand_joints   = hf['hand_joints_gt'][()]
        hand_contacts = hf['hand_contacts_gt'][()]
    else:
        h5_file = f'{second_path}/data/hdf5/{category}/{basename[:-2]}/{basename[-1]}.h5'
        with h5py.File(h5_file, 'r') as handle:
            hand_joints   = handle['joints'][()]
            hand_contacts = handle['contacts'][()]
    hand_trans = np.mean(hand_joints - joints_canon[0], axis=0).reshape(1, 3) # use GT joints

    # >>> get pred
    return [hand_trans, hand_trans]

def get_parts_nocs(hf, part_idx_list_gt, part_idx_list_pred, nocs='part', num_parts=4, verbose=False):
    nocs_err = []
    scale_list = []
    rt_list  = []
    nocs_gt        =  {}
    nocs_pred      =  {}
    nocs_gt['pn']  =  hf['nocs_per_point_gt'][()].transpose(1, 0)
    nocs_pred['pn']=  hf['nocs_per_point_pred'][()].transpose(1, 0)
    for j in range(num_parts-1):
        if nocs == 'part':
            a = nocs_gt['pn'][part_idx_list_gt[j], :]
            b1 = nocs_pred['pn'][part_idx_list_pred[j], 3*j:3*(j+1)]
            b = nocs_pred['pn'][part_idx_list_gt[j], 3*j:3*(j+1)]
        else:
            a = nocs_gt['gn'][part_idx_list_gt[j], :]
            if nocs_pred['gn'].shape[1] ==3:
                b = nocs_pred['gn'][part_idx_list_gt[j], :3]
            else:
                b = nocs_pred['gn'][part_idx_list_gt[j], 3*j:3*(j+1)]
        c = input_pts[part_idx_list_gt[j], :]
        c1 = input_pts[part_idx_list_pred[j], :]
        nocs_err.append(np.mean(np.linalg.norm(a - b, axis=1)))
        print('')
        print(f'Part {j} GT nocs: \n', b)
        print(f'Part {j} Pred nocs: \n', b)
        if verbose:
            plot3d_pts([[a, b]], [['GT part {}'.format(j), 'Pred part {}'.format(j)]], s=5**2, title_name=['NPCS comparison'], limits = [[0, 1], [0, 1], [0, 1]])
            plot3d_pts([[a], [a]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=5**2, title_name=['GT NPCS', 'Pred NPCS'], color_channel=[[a], [b]], save_fig=True, limits = [[0, 1], [0, 1], [0, 1]], sub_name='{}'.format(j))
            plot3d_pts([[a]], [['Part {}'.format(j)]], s=5**2, title_name=['NOCS Error'], color_channel=[[np.linalg.norm(a - b, axis=1)]], limits = [[0, 1], [0, 1], [0, 1]], colorbar=True)
            plot3d_pts([[a], [c]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=5**2, title_name=['GT NOCS', 'point cloud'],limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], color_channel=[[a], [a]], save_fig=True, sub_name='{}'.format(j))
        s, r, t, rt = estimateSimilarityUmeyama(a.transpose(), c.transpose())
        rt_list.append(compose_rt(r, t))
        scale_list.append(s)
        s, r, t, rt = estimateSimilarityUmeyama(b1.transpose(), c1.transpose())
        rt_list.append(compose_rt(r, t))
        scale_list.append(s)
    # if args.viz and j==3:
    #     d = np.matmul(a*s, r) + t
    #     plot3d_pts([[d, c]], [['Part {} align'.format(j), 'Part {} cloud'.format(j)]], s=15, title_name=['estimation viz'], color_channel=[[a, a]], save_fig=True, sub_name='{}'.format(j))
    return nocs_gt, nocs_pred

def get_object_pose(hf, basename, nocs_list, pts_list):
    RTS=[]

    return RTS

def get_contacts_camera_raw(hf, basename=None, category='eyeglasses', verbose=False):
    #>>>>>>>>>>>>>>>>>>>> get gt
    if 'hand_contacts_gt' in list(hf.keys()):
        hand_contacts = hf['hand_contacts_gt'][()]
    else:
        h5_file = f'{second_path}/data/hdf5/{category}/{basename[:-2]}/{basename[-1]}.h5'
        with h5py.File(h5_file, 'r') as handle:
            hand_contacts = handle['contacts'][()]

    #>>>>>>>>>>>>>> get preds
    hdf5_file = None
    # read contacts predictions

    return [hand_contacts, hand_contacts]

def get_contacts_camera(hf, basename, verbose=False):
    # box_label_mask_gt (64,)
    # center_label_gt (64, 3)

    # center_pred
    # object_assignment_pred (128,)
    # objectness_label_pred (128,)
    # objectness_mask_pred (128,)
    # objectness_scores_pred (128, 2)
    coords = hf['center_label_gt'][()]
    mask   = hf['box_label_mask_gt'][()]
    coords_gt = coords[np.where(mask>0)[0], :]

    coords =  hf['center_pred'][()]
    mask   =  hf['objectness_label_pred'][()]
    scores = F.softmax(torch.FloatTensor(hf['objectness_scores_pred'][()]), dim=1).cpu().numpy()
    valid_ind = scores[:, 1].argsort()[-10:][::-1]
    # coords_pred = coords[np.where(mask>0.1)[0], :]
    coords_pred = coords[valid_ind, :]

    if verbose:
        plot3d_pts([[coords_gt], [coords_gt, coords_pred]], [['GT'], ['GT', 'Pred']], s=2**2, mode='continuous', dpi=200, title_name=['contact pts', 'contact pts'])
    return [coords_gt, coords_pred]

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
    # breakpoint()
    if verbose:
        plot_arrows(input_pts[ind_vote], [offset_gt[ind_vote], offset_pred], whole_pts=input_pts, title_name='contact voting', dpi=200, s=1, thres_r=0.2, sparse=True)

def plot_hand_skeleton(hand_joints):
    pass

def breakpoint():
    import pdb;pdb.set_trace()

def check_h5_keys(hf):
    print('')
    for key in list(hf.keys()):
        print(key, hf[key][()].shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--domain', default='seen', help='which sub test set to choose')
    parser.add_argument('--exp_num', default='0.8', required=True) # default is 0.3
    parser.add_argument('--nocs', default='part', help='which sub test set to choose')

    parser.add_argument('--hand', action='store_true', help='whether to visualize hand')
    parser.add_argument('--vote', action='store_true', help='whether to vote hand joints')
    parser.add_argument('--mano', action='store_true', help='whether to use mano')
    parser.add_argument('--save', action='store_true', help='save err to pickles')
    parser.add_argument('--save_fig', action='store_true', help='save err to pickles')
    parser.add_argument('--show_fig', action='store_true', help='save err to pickles')
    parser.add_argument('--viz', action='store_true', help='whether to viz')

    args = parser.parse_args()

    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    unseen_instances= dset_info.test_list
    special_ins     = dset_info.spec_list
    main_exp        = args.exp_num
    root_dset       = second_path + '/data'

    if args.domain == 'demo':
        base_path       = second_path + '/results/demo'
        test_h5_path    = base_path + '/{}'.format(main_exp)
        test_group      = get_demo_h5(os.listdir(test_h5_path))
    else:
        base_path       = second_path + '/model' # testing
        print('---checking results from ', base_path)
        test_h5_path    = base_path + f'/{args.item}/{args.exp_num}/preds/{args.domain}'
        test_group      = get_full_test(os.listdir(test_h5_path), unseen_instances, domain=args.domain, spec_instances=special_ins)

    print('---we have {} testing data for {} {}'.format(len(test_group), args.domain, args.item))

    save_path = base_path + f'/pickle/{args.item}/{args.exp_num}'
    file_name = base_path + f'/pickle/{args.item}/{args.exp_num}/{args.domain}_hand.pkl'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # start_time = time.time()
    # pca_hand = np.load(second_path + f'/data/pickle/hand_pca.npy',allow_pickle=True).item()
    # mean_u  = pca_hand['M'] # 63
    # eigen_v = pca_hand['E'] # 30*63
    error_all = []

    i = 0
    # for i in range(len(test_group)):
    h5_files   = []
    hf_list    = []
    h5_file    =  test_h5_path + '/' + test_group[i]
    h5_files.append(h5_file)
    h5_file    =  h5_file.replace('0.8', '0.9')
    h5_files.append(h5_file)
    print('')
    print('----------Now checking {}: {}'.format(i, h5_file))
    for h5_file in h5_files:
        hf             =  h5py.File(h5_file, 'r')
        basename       =  hf.attrs['basename']
        check_h5_keys(hf)
        hf_list.append(hf)


    hf = hf_list[1]
    basename       =  hf.attrs['basename']

    # 1. get pts and gt contacts
    instance, art_index, grasp_ind, frame_order = basename.split('_')[0:4]
    part_idx_list_gt, part_idx_list_pred = get_parts_ind(hf, num_parts=4)
    input_pts, gt_parts, pred_parts      = get_parts_pcloud(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4, verbose=False)
    contact_pts = get_contacts_camera(hf, basename, verbose=True)
    plot3d_pts([[input_pts, contact_pts[0], contact_pts[1]]], [['input', 'GT contact', 'Pred contact']], s=2, mode='continuous',  limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 1.5]], title_name=['input + contact pts'])

    # 2. check gt vote offset
    get_contacts_vote(hf, basename, input_pts, verbose=True)
    # combined visualization
    plt.show()
    plt.close()
    #
    # hf = hf_list[0]
    # instance, art_index, grasp_ind, frame_order = basename.split('_')[0:4]
    # part_idx_list_gt, part_idx_list_pred = get_parts_ind(hf, num_parts=4)
    # input_pts, gt_parts, pred_parts      = get_parts_pcloud(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4, verbose=False)
    # nocs_gt, nocs_pred                   = get_parts_nocs(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4)
    #
    # if args.hand:
    #     if args.vote:
    #         results = vote_hand_joints_cam(hf, input_pts, part_idx_list_gt, verbose=False)
    #     if args.mano:
    #         verts_canon, faces_canon, joints_canon = get_hand_mesh_canonical(hf) # in hand canonical space
    #     else:
    #         joints_cam = get_hand_regression_camera(hf, input_pts, verbose=False)
    #
    #     contact_pts = get_contacts_camera_raw(hf, basename, verbose=False)
    #     hand_trans  = get_hand_trans2camera(hf, basename, joints_canon, category=args.item, verbose=False)
    #     verts_cam, faces_cam, joints_cam = get_hand_mesh_camera(verts_canon, faces_canon, joints_canon, hand_trans)
    #
    #     mean_hand = hf['mean_hand_gt'][()]
    #     # hand pose visualization
    #     plot_hand_w_object(obj_verts=verts_cam[1], obj_faces=faces_cam[1], hand_verts=verts_cam[0], hand_faces=faces_cam[0], s=5, pts=[[contact_pts[0], gt_parts[-1]+mean_hand]], save=False, mode='continuous')
    #
    #     # object poses estimation
    #     obj_vertices, obj_faces = get_obj_mesh(basename, verbose=False)
    #     obj_vertices_tf         = transform_pcloud(np.copy(obj_vertices), RT=hf['extrinsic_params_gt'][()], extra_R=None)
    #     # compute object pose
    #     RTS = get_object_pose(hf, basename, [nocs_gt, nocs_pred], [gt_parts, pred_parts])
    #
    #     # contacts visualization
    #     plot_hand_w_object(obj_verts=obj_vertices_tf, obj_faces=obj_faces, hand_verts=verts_cam[0], hand_faces=faces_cam[0], s=20, pts=[[contact_pts[0], gt_parts[-1]+mean_hand]], save=False, mode='continuous')
    #     plot_hand_w_object(obj_verts=obj_vertices_tf, obj_faces=obj_faces, hand_verts=verts_cam[1], hand_faces=faces_cam[1], s=20, pts=[[contact_pts[0], gt_parts[-1]+mean_hand]], save=False, mode='continuous')
    #     # compute gt offsets
    #     #
    #     # combined visualization
    #     plt.show()
    #     plt.close()
    #
    #     error_dist = joints_cam[1] - joints_cam[0]
    #     error_all.append(error_dist) # J, 3
    # error_norm = np.mean( np.linalg.norm(np.stack(error_all, axis=0), axis=2), axis=0)
    # print('experiment: ', args.exp_num)
    # for j, err in enumerate(error_norm):
    #     print(err)
    # print('done!')
