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

def print_group(names, values):
    for subname, subvalue in zip(names, values):
        print(subname, ':\n', subvalue)
        print('')
# np.savetxt(f'{save_path}/contact_err_{domain}.txt', contacts_err, fmt='%1.5f', delimiter='\n')

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
        plot3d_pts([[input_pts]], [['input']], s=3**2, title_name=['inputs'], sub_name=str(i), axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([gt_parts], [['Part {}'.format(j) for j in range(num_parts)]], s=3**2, title_name=['GT seg'], sub_name=str(i),  axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([gt_parts[:3]], [['Part {}'.format(j) for j in range(3)]], s=3**2, title_name=['GT object'], sub_name=str(i),  axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([pred_parts], [['Part {}'.format(j) for j in range(num_parts)]], s=3**2, title_name=['Pred seg'], sub_name=str(i), axis_off=False, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)

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

    if verbose:
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
        plot_hand_w_object(obj_verts=verts[0], obj_faces=faces[0], hand_verts=verts[1], hand_faces=faces[1], s=5**2, save=False)

    return verts, faces, joints

def get_hand_mesh_camera(verts_canon, faces_canon, joints_canon, hand_trans, verbose=False):
    verts, faces, joints = [], [], []
    for j in range(2):
        verts.append(verts_canon[j] + hand_trans[j])
        joints.append(joints_canon[j] + hand_trans[j])
    faces = faces_canon
    if verbose:
        plot_hand_w_object(obj_verts=verts[0], obj_faces=faces[0], hand_verts=verts[1], hand_faces=faces[1], s=5**2, mode='continuous', save=False)

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
def get_hand_trans2camera(hf, basename, joints_canon, extra_hf=None, category='eyeglasses', verbose=False):
    #>>>>>>>>>>>>>>>>>>>> get gt
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
        if verbose:
            print(f'Part {j} GT nocs: \n', b)
            print(f'Part {j} Pred nocs: \n', b)
            plot3d_pts([[a, b]], [['GT part {}'.format(j), 'Pred part {}'.format(j)]], s=5**2, title_name=['NPCS comparison'], limits = [[0, 1], [0, 1], [0, 1]])
            plot3d_pts([[a], [a]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=5**2, title_name=['GT NPCS', 'Pred NPCS'], color_channel=[[a], [b]], save_fig=True, limits = [[0, 1], [0, 1], [0, 1]], sub_name='{}'.format(j))
            plot3d_pts([[a]], [['Part {}'.format(j)]], s=5**2, title_name=['NOCS Error'], color_channel=[[np.linalg.norm(a - b, axis=1)]], limits = [[0, 1], [0, 1], [0, 1]], colorbar=True)
            plot3d_pts([[a], [c]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=5**2, title_name=['GT NOCS', 'point cloud'],limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], color_channel=[[a], [a]], save_fig=True, sub_name='{}'.format(j))
        # s, r, t, rt = estimateSimilarityUmeyama(a.transpose(), c.transpose())
        # rt_list.append(compose_rt(r, t))
        # scale_list.append(s)
        # s, r, t, rt = estimateSimilarityUmeyama(b1.transpose(), c1.transpose())
        # rt_list.append(compose_rt(r, t))
        # scale_list.append(s)
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
        plot3d_pts([[coords_gt], [coords_gt, coords_pred]], [['GT'], ['GT', 'Pred']], s=[10**2, 5**2, 5**2], mode='continuous', dpi=200, title_name=['contact pts', 'Objectness'], color_channel=[[gt_color], [gt_color, objectness_color]])
        plot3d_pts([[coords_gt], [coords_gt, coords_pred]], [['GT'], ['GT', 'Pred']], s=[10**2, 5**2, 5**2], mode='continuous', dpi=200, title_name=['contact pts', 'Confidence'], color_channel=[[gt_color], [gt_color, confidence_color]])
        plot3d_pts([[coords_gt], [coords_gt, coords_pred_final]], [['GT'], ['GT', 'Pred']], s=[10**2, 5**2, 5**2], mode='continuous', dpi=200, title_name=['contact pts', 'Final'], color_channel=[[gt_color], [gt_color, confidence_color_final]])

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

def plot_hand_skeleton(hand_joints):
    pass

def breakpoint():
    import pdb;pdb.set_trace()

def check_h5_keys(hf, verbose=False):
    print('')
    if verbose:
        for key in list(hf.keys()):
            print(key, hf[key][()].shape)

def get_pca_stuff(second_path):
    pca_hand = np.load(second_path + f'/data/pickle/hand_pca.npy',allow_pickle=True).item()
    mean_u  = pca_hand['M'] # 63
    eigen_v = pca_hand['E'] # 30*63
    return eigen_v, mean_u

def add_contacts_evaluation(stat_dict, contact_pts, contacts_num):
    stat_dict['gt'].append(contact_pts[0])
    stat_dict['pred'].append(contact_pts[1])
    stat_dict['gt_cnt']+=contacts_num[0]
    stat_dict['pred_cnt']+=contacts_num[1]

def eval_contacts(domain, contacts_stat, save_path):
    contacts_err = np.linalg.norm(np.concatenate(contacts_stat['gt'], axis=0) - np.concatenate(contacts_stat['pred'], axis=0), axis=1)
    mean_contacts_err = np.mean(contacts_err)
    contacts_miou    = contacts_stat['pred_cnt']/contacts_stat['gt_cnt']
    np.savetxt(f'{save_path}/contact_err_{domain}.txt', contacts_err, fmt='%1.5f', delimiter='\n')

    print(f'>>>>>>>>>>>>>   {domain}    <<<<<<<<<<<< \n contacts_stat:\n contacts_err: {contacts_err}\n mean_contacts_err: {mean_contacts_err}\n contacts_miou:{contacts_miou}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--domain', default='seen', help='which sub test set to choose')
    parser.add_argument('--exp_num', default='0.8', required=False) # default is 0.3
    parser.add_argument('--nocs', default='part', help='which sub test set to choose')

    parser.add_argument('--hand', action='store_true', help='whether to visualize hand')
    parser.add_argument('--contact', action='store_true', help='whether to visualize hand')
    parser.add_argument('--vote', action='store_true', help='whether to vote hand joints')
    parser.add_argument('--mano', action='store_true', help='whether to use mano')
    parser.add_argument('--save', action='store_true', help='save err to pickles')
    parser.add_argument('--save_fig', action='store_true', help='save err to pickles')
    parser.add_argument('--show_fig', action='store_true', help='save err to pickles')
    parser.add_argument('--viz', action='store_true', help='whether to viz')
    parser.add_argument('--verbose', action='store_true', help='whether to viz')

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

    start_time = time.time()
    error_all = []
    contacts_stat = {'gt': [], 'pred': [], 'gt_cnt': 0, 'pred_cnt': 0}
    exp_nums = ['0.94', args.exp_num, '1.1'] # contacts, mano, translation
    for i in range(len(test_group)):
        h5_files   = []
        hf_list    = []
        h5_file    =  test_h5_path + '/' + test_group[i]
        h5_files.append( h5_file.replace(args.exp_num, exp_nums[0]))
        h5_files.append( h5_file.replace(args.exp_num, exp_nums[1]))
        h5_files.append( h5_file.replace(args.exp_num, exp_nums[2]))

        print('')
        print('----------Now checking {}: {}'.format(i, h5_file))
        for h5_file in h5_files:
            hf             =  h5py.File(h5_file, 'r')
            basename       =  hf.attrs['basename']
            check_h5_keys(hf, verbose=args.verbose)
            hf_list.append(hf)

        if args.contact:
            hf = hf_list[0]
            basename       =  hf.attrs['basename']
            # 1. get pts and gt contacts
            instance, art_index, grasp_ind, frame_order = basename.split('_')[0:4]
            part_idx_list_gt, part_idx_list_pred = get_parts_ind(hf, num_parts=4)
            input_pts, gt_parts, pred_parts      = get_parts_pcloud(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4, verbose=False)
            contact_pts, contacts_num = get_contacts_camera(hf, basename, verbose=args.verbose)
            add_contacts_evaluation(contacts_stat, contact_pts, contacts_num)
            if args.verbose:
                plot3d_pts([[input_pts, contact_pts[0], contact_pts[1]]], [['input', 'GT contact', 'Pred contact']], s=[1**2, 10**2, 10**2], mode='continuous',  limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 1.5]], title_name=['input + contact pts'])

            # 2. check gt vote offset
            get_contacts_vote(hf, basename, input_pts, verbose=args.verbose)

            # combined visualization
            if args.viz:
                plt.show()
                plt.close()

        hf = hf_list[1]
        instance, art_index, grasp_ind, frame_order = basename.split('_')[0:4]
        part_idx_list_gt, part_idx_list_pred = get_parts_ind(hf, num_parts=4)
        input_pts, gt_parts, pred_parts      = get_parts_pcloud(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4, verbose=False)
        nocs_gt, nocs_pred                   = get_parts_nocs(hf, part_idx_list_gt, part_idx_list_pred, num_parts=4, verbose=False)

        if args.hand:
            if args.vote:
                results = vote_hand_joints_cam(hf, input_pts, part_idx_list_gt, verbose=args.verbose)
            if args.mano:
                verts_canon, faces_canon, joints_canon = get_hand_mesh_canonical(hf, verbose=False) # in hand canonical space
            else:
                hand_joints_cam  = get_hand_regression_camera(hf, input_pts, verbose=args.verbose)

            contact_pts = get_contacts_camera_raw(hf, basename, verbose=args.verbose)
            hand_trans  = get_hand_trans2camera(hf, basename, joints_canon, extra_hf=hf_list[2], category=args.item, verbose=False)

            hand_vertices_cam, hand_faces_cam, hand_joints_cam  = get_hand_mesh_camera(verts_canon, faces_canon, joints_canon, hand_trans, verbose=False)

            mean_hand = hf['mean_hand_gt'][()]

            # object poses estimation

            # obj_vertices, obj_faces = get_obj_mesh(basename, verbose=args.verbose)
            # obj_vertices_tf         = transform_pcloud(np.copy(obj_vertices), RT=hf['extrinsic_params_gt'][()], extra_R=None)
            #
            # # cononical
            # refer_path = '/home/dragon/Documents/CVPR2020/dataset/shape2motion/objects/eyeglasses/{}/part_objs/{}.obj'
            # name_list = ['none_motion', 'dof_rootd_Aa002_r', 'dof_rootd_Aa001_r']
            # if args.verbose:
            #     for part_name in name_list:
            #         part_vertices, part_faces = get_obj_mesh(basename, full_path=refer_path.format(instance, part_name), verbose=True)

            # compute object pose
            RTS = get_object_pose(hf, basename, [nocs_gt, nocs_pred], [gt_parts, pred_parts])

            # contacts visualization
            if args.verbose:
            #     # hand
                # plot_hand_w_object(obj_verts=hand_vertices_cam[1], obj_faces=hand_faces_cam[1], hand_verts=hand_vertices_cam[0], hand_faces=hand_faces_cam[0], s=5, pts=[[contact_pts[0], gt_parts[-1]+mean_hand]], save=False, mode='continuous')
            #     # print_group(['obj_vertices_tf', 'obj_faces', 'hand_vertices_cam[0]', 'hand_faces_cam[0]',  'contact_pts', 'gt_parts'], [obj_vertices_tf.shape, obj_faces.shape, hand_vertices_cam[0].shape, hand_faces_cam[0].shape, contact_pts.shape, gt_parts[-1].shape])
                plot_hand_w_object(obj_verts=obj_vertices_tf, obj_faces=obj_faces, hand_verts=obj_vertices_tf, hand_faces=obj_faces, s=20, save=False, mode='continuous')
                # plot_hand_w_object(obj_verts=obj_vertices_tf, obj_faces=obj_faces, hand_verts=obj_vertices_tf, hand_faces=obj_faces, s=20, pts=[[contact_pts[0], gt_parts[-1]+mean_hand]], save=False, mode='continuous')
            #     plot_hand_w_object(obj_verts=obj_vertices_tf, obj_faces=obj_faces, hand_verts=hand_vertices_cam[1], hand_faces=hand_faces_cam[1], s=20, pts=[[contact_pts[0], gt_parts[-1]+mean_hand]], save=False, mode='continuous')
            #
            # combined visualization
            if args.viz:
                plt.show()
                plt.close()

            error_dist = hand_joints_cam[1] - hand_joints_cam[0]
            error_all.append(error_dist) # J, 3
    if args.contact:
        eval_contacts(args.domain, contacts_stat, save_path)

    if args.hand:
        error_norm = np.mean( np.linalg.norm(np.stack(error_all, axis=0), axis=2), axis=0)
        print('experiment: ', args.exp_num)
        for j, err in enumerate(error_norm):
            print(err)
    end_time = time.time()
    print(f'---it takes {end_time-start_time} seconds')
