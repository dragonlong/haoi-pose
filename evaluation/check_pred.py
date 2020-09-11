import numpy as np
import os
import sys
import time
import json
import h5py
import pickle
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

def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation.transpose()
    aligned_RT[:3, 3]  = translation
    aligned_RT[3, 3]   = 1
    return aligned_RT

def breakpoint():
    import pdb;pdb.set_trace()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='unseen', help='which sub test set to choose')
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
        base_path       = second_path + '/results/test_pred' # testing
        test_h5_path    = base_path + '/{}'.format(main_exp)
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
        if args.hand: 
            hand_heatmap_gt = hf['hand_heatmap_gt'][()]
            hand_unitvec_gt = -hf['hand_unitvec_gt'][()]
            hand_heatmap_pred   = hf['hand_heatmap_per_point'][()]
            hand_unitvec_pred   = -hf['hand_unitvec_per_point'][()]
            thres_r       = 0.2
            for j in range(21):
                offset_gt  = hand_unitvec_gt[:, 3*j:3*(j+1)] * (1- hand_heatmap_gt[:, j:j+1].reshape(-1, 1)) * thres_r
                offset_pred  = hand_unitvec_pred[:, 3*j:3*(j+1)] * (1- hand_heatmap_pred[:, j:j+1].reshape(-1, 1)) * thres_r
                idx = np.where(mask_gt==3)[0]
                plot_arrows(input_pts[idx], [offset_gt[idx], offset_pred[idx]], whole_pts=input_pts, title_name='{}th_joint_voting'.format(j), limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], dpi=200, s=25, thres_r=0.2, sparse=True, save=args.save_fig, index=i)

        part_idx_list_gt   = []
        part_idx_list_pred   = []
        cls_per_pt_pred= np.argmax(mask_pred, axis=1)
        nocs_err = []
        rt_gt    = []
        scale_gt = []
        if args.viz:
            plot3d_pts([[input_pts]], [['']], s=2**2, title_name=['Input pts'], limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], axis_off=True)
        for j in range(num_parts):
            part_idx_list_gt.append(np.where(mask_gt==j)[0])
            part_idx_list_pred.append(np.where(cls_per_pt_pred==j)[0])

            if args.nocs == 'part':
                a = nocs_gt['pn'][part_idx_list_gt[j], :]
                b = nocs_pred['pn'][part_idx_list_gt[j], 3*j:3*(j+1)]
            else:
                a = nocs_gt['gn'][part_idx_list_gt[j], :]
                if nocs_pred['gn'].shape[1] ==3:
                    b = nocs_pred['gn'][part_idx_list_gt[j], :3]
                else:
                    b = nocs_pred['gn'][part_idx_list_gt[j], 3*j:3*(j+1)]
            c = input_pts[part_idx_list_gt[j], :]
            nocs_err.append(np.mean(np.linalg.norm(a - b, axis=1)))
            if args.viz and j==3:
                plot3d_pts([[a, b]], [['GT part {}'.format(j), 'Pred part {}'.format(j)]], s=10**2, title_name=['NPCS comparison'], limits = [[0, 1], [0, 1], [0, 1]])
                plot3d_pts([[a], [a]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=10**2, title_name=['GT NPCS', 'Pred NPCS'], color_channel=[[a], [b]], save_fig=True, limits = [[0, 1], [0, 1], [0, 1]], sub_name='{}'.format(j))
                plot3d_pts([[a]], [['Part {}'.format(j)]], s=10**2, title_name=['NOCS Error'], color_channel=[[np.linalg.norm(a - b, axis=1)]], limits = [[0, 1], [0, 1], [0, 1]], colorbar=True)
                plot3d_pts([[a], [c]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=10**2, title_name=['GT NOCS', 'point cloud'],limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], color_channel=[[a], [a]], save_fig=True, sub_name='{}'.format(j))
            s, r, t, rt = estimateSimilarityUmeyama(a.transpose(), c.transpose())
            rt_gt.append(compose_rt(r, t))
            scale_gt.append(s)
            # if args.viz and j==3:
            #     d = np.matmul(a*s, r) + t
            #     plot3d_pts([[d, c]], [['Part {} align'.format(j), 'Part {} cloud'.format(j)]], s=15, title_name=['estimation viz'], color_channel=[[a, a]], save_fig=True, sub_name='{}'.format(j))
        
        plot3d_pts([[input_pts[part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=10**2, title_name=['GT seg'], sub_name=str(i),  axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        plot3d_pts([[input_pts[part_idx_list_pred[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=10**2, title_name=['Pred seg'], sub_name=str(i), axis_off=True, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
        # thres_r       = 0.2
        # offset        = unitvec_gt * (1- heatmap_gt.reshape(-1, 1)) * thres_r
        # joint_pts     = nocs_gt['gn'] + offset
        # joints_list   = []
        # idx           = np.where(joint_cls_gt > 0)[0]
        # plot_arrows(nocs_gt['gn'][idx], [0.5*orient_gt[idx]], whole_pts=nocs_gt['gn'], title_name='{}_joint_pts_axis'.format('GT'), dpi=200, s=25, thres_r=0.2, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
        # plot_arrows(nocs_gt['gn'][idx], [offset[idx]], whole_pts=nocs_gt['gn'], title_name='{}_joint_offsets'.format('GT'), dpi=200, s=25, thres_r=0.2, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
        # # plot_arrows(input_pts[idx], [0.5*orient_gt[idx]], whole_pts=input_pts, title_name='{}_joint_pts_axis'.format('GT'), dpi=200, s=25, thres_r=0.2, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
        # # plot_arrows(input_pts[idx], [offset[idx]], whole_pts=input_pts, title_name='{}_joint_offsets'.format('GT'), dpi=200, s=25, thres_r=0.2, show_fig=args.show_fig, sparse=True, save=args.save_fig, index=i, save_path=root_dset + '/NOCS/' + args.item)
   

        rts_dict   = {}
        scale_dict = {}
        rt_dict    = {}
        rt_dict['gt']    = rt_gt
        scale_dict['gt'] = scale_gt
        rts_dict['scale']  = scale_dict
        rts_dict['rt']     = rt_dict
        rts_dict['nocs_err'] = nocs_err
        all_rts[basename]  = rts_dict
        # except:
        #     # print('skipping ', test_group[i])
        #     pass

    # if args.save:
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(all_rts, f, protocol=2)
    #         print('saving to ', file_name)
    # # print('takes {} seconds', time.time() - start_time)

    # # plot3d_pts([[input_pts[part_idx_list_gt[0], :], input_pts[part_idx_list_gt[1], :], input_pts[part_idx_list_gt[2], :]]], [['Part {}'.format(j) for j in range(3)]], s=5, title_name=['GT Seg'])
    # # plot3d_pts([[input_pts[part_idx_list_pred[0], :], input_pts[part_idx_list_pred[1], :], input_pts[part_idx_list_pred[2], :]]], [['Part {}'.format(j) for j in range(3)]], s=5, title_name=['Pred Seg'])
    # # plot3d_pts([[nocs_gt[part_idx_list_pred[0], :], nocs_gt[part_idx_list_pred[1], :], nocs_gt[part_idx_list_pred[2], :]]], [['Part {}'.format(j) for j in range(3)]], s=5, title_name=['GT NOCS'])
    # # plot3d_pts([[nocs_pred[part_idx_list_pred[0], 3:6], nocs_pred[part_idx_list_pred[1], 0:3], nocs_pred[part_idx_list_pred[2], 6:9]]], [['Part {}'.format(j) for j in range(3)]], s=5, title_name=['Pred NOCS'])
    # # plot3d_pts([[input_pts]], [['']], s=5, title_name=['NOCS Confidence'], color_channel=[[np.concatenate([confidence, np.zeros((confidence.shape[0], 2))], axis=1)]])
