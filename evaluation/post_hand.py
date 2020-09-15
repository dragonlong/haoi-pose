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

import _init_paths
from common.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list
from common.data_utils import get_demo_h5, get_full_test
from common.aligning import estimateSimilarityTransform, estimateSimilarityUmeyama
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


def breakpoint():
    import pdb;pdb.set_trace()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='seen', help='which sub test set to choose')
    parser.add_argument('--nocs', default='part', help='which sub test set to choose')
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--save', action='store_true', help='save err to pickles')
    parser.add_argument('--save_fig', action='store_true', help='save err to pickles')
    parser.add_argument('--show_fig', action='store_true', help='save err to pickles')
    parser.add_argument('--viz', action='store_true', help='whether to viz')
    parser.add_argument('--hand', action='store_true', help='whether to visualize hand')
    args = parser.parse_args()


    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    unseen_instances= dset_info.test_list
    special_ins     = dset_info.spec_list
    main_exp        = dset_info.joint_baseline
    baseline_exp    = dset_info.baseline
    root_dset       = second_path + '/data'

    if args.domain == 'demo':
        base_path       = second_path + '/results/demo'
        test_h5_path    = base_path + '/{}'.format(main_exp)
        test_group      = get_demo_h5(os.listdir(test_h5_path))
    else:
        base_path       = my_dir + '/results/val_pred' # testing
        test_h5_path    = base_path + '/{}/{}'.format(main_exp, 'step140000')
        test_group      = get_full_test(os.listdir(test_h5_path), unseen_instances, domain=args.domain, spec_instances=special_ins)

    all_bad             = []
    print('we have {} testing data for {} {}'.format(len(test_group), args.domain, args.item))
    all_rts = {}
    file_name = base_path + '/pickle/{}/{}_{}_{}_rt.pkl'.format(main_exp, args.domain, args.nocs, args.item)
    save_path = base_path + '/pickle/{}/'.format(main_exp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    problem_ins = []
    start_time = time.time()

    #
    pca_hand = np.load(second_path + f'/data/pickle/hand_pca.npy',allow_pickle=True).item()
    mean_u  = pca_hand['M'] # 63
    eigen_v = pca_hand['E'] # 30*63
    error_all = []
    for i in range(len(test_group)):
        # try:
        h5_file    =  test_h5_path + '/' + test_group[i]
        print('Now checking {}: {}'.format(i, h5_file))
        if test_group[i].split('_')[0] in problem_ins:
            print('skipping ', test_group[0][i])
            continue
        hf             =  h5py.File(h5_file, 'r')
        basename       =  hf.attrs['basename']

        nocs_gt        =  {}
        nocs_pred      =  {}
        nocs_gt['pn']  =  hf['nocs_gt'][()]
        nocs_pred['pn']=  hf['nocs_per_point'][()]

        if args.nocs == 'global':
            nocs_gt['gn']  =  hf['nocs_gt_g'][()]
            if args.domain == 'demo':
                nocs_pred['gn']=  nocs_gt['gn']
            else:
                nocs_pred['gn']=  hf['gocs_per_point'][()]
        input_pts      =  hf['P'][()]
        mask_gt        =  hf['cls_gt'][()]
        name_info      = basename.split('_')
        instance       = name_info[0]
        art_index      = name_info[1]
        grasp_ind      = name_info[2]
        frame_order    = name_info[3]
        mask_pred      =  hf['instance_per_point'][()]
        if args.viz:
            plot3d_pts([[input_pts]], [['']], s=2**2, title_name=['Input pts'], limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], axis_off=True)

        if args.hand:
            hand_heatmap_gt     = hf['hand_heatmap_gt'][()]
            hand_unitvec_gt     = -hf['hand_unitvec_gt'][()] # todo
            hand_heatmap_pred   = hf['hand_heatmap_per_point'][()]
            hand_unitvec_pred   = -hf['hand_unitvec_per_point'][()]
            thres_r       = 0.2

            thres_var     = 0.075
            results = {'vote': [], 'divergence': [], 'regression':None, 'final': []}
            for j in range(21):
                offset_gt  = hand_unitvec_gt[:, 3*j:3*(j+1)] * (1- hand_heatmap_gt[:, j:j+1].reshape(-1, 1)) * thres_r
                offset_pred  = hand_unitvec_pred[:, 3*j:3*(j+1)] * (1- hand_heatmap_pred[:, j:j+1].reshape(-1, 1)) * thres_r
                idx = np.where(np.argmax(mask_pred, axis=1)==3)[0]
                ind_vote = np.where(hand_heatmap_pred[idx, j:j+1] > 0.1)[0]
                if len(ind_vote) ==0:
                    results['vote'].append(np.array([0, 0, 0]).reshape(1, 3))
                    results['divergence'].append(1)
                    continue
                joint_votes = input_pts[idx][ind_vote] + offset_pred[idx][ind_vote]
                vote_var         = np.sum(np.var(joint_votes, axis=0)) # N, 3
                joint_voted      = np.mean(joint_votes, axis=0).reshape(1, 3)
                results['vote'].append(joint_voted)
                results['divergence'].append(vote_var)
                if args.viz:
                    plot_arrows(input_pts[idx], [offset_gt[idx], offset_pred[idx]], whole_pts=input_pts, title_name='{}th_joint_voting'.format(j), limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], dpi=200, s=25, thres_r=0.2, sparse=True, save=args.save_fig, index=i)

            # hand joints regression
            hand_joints_gt   = hf['hand_joints_gt'][()].reshape(-1, 3)
            hand_joints_pred = hf['hand_joints_pred'][()].reshape(-1, 3)
            # breakpoint()
            results['regression'] = hand_joints_pred
            results['gt']         = hand_joints_gt

            for j in range(21):
                diver = results['divergence'][j]

                if diver > thres_var:
                    results['final'].append(hand_joints_pred[j:j+1,:])
                else:
                    results['final'].append(results['vote'][j])

            pred_flatten = np.concatenate(results['final']).reshape(-1)
            pred_hands   =  (np.dot( np.dot(eigen_v.T, eigen_v), (pred_flatten-mean_u).reshape(-1, 1)) + mean_u.reshape(-1, 1)).reshape(-1, 3)

            error_dist = pred_hands.reshape(-1, 3) - results['gt']
            error_dist[np.where(error_dist>1)[0], np.where(error_dist>1)[1]] = 0
            error_dist[np.where(error_dist<-1)[0], np.where(error_dist<-1)[1]] = 0
            error_all.append(error_dist) # J, 3

        # part_idx_list_gt     = []
        # part_idx_list_pred   = []
        # cls_per_pt_pred= np.argmax(mask_pred, axis=1)
        # nocs_err = []
        # rt_gt    = []
        # scale_gt = []

        # for j in range(num_parts):
        #     part_idx_list_gt.append(np.where(mask_gt==j)[0])
        #     part_idx_list_pred.append(np.where(cls_per_pt_pred==j)[0])

    # get hand_joints regression results


    # get hand offsets, input points, GT hand


    # check variance

    # evaluate mean joint error
    breakpoint()
    error_norm = np.mean( np.linalg.norm(np.stack(error_all, axis=0), axis=2), axis=0)
    print(error_norm.shape, '\n', error_norm)
    print('ah!')
