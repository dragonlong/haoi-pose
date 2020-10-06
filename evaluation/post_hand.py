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
from sklearn.decomposition import PCA
from manopth import rodrigues_layer
from manopth.manolayer import ManoLayer
import torch
import _init_paths
from common.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list, plot_hand_w_object
from common.data_utils import get_demo_h5, get_full_test, save_objmesh, fast_load_obj
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
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--domain', default='seen', help='which sub test set to choose')
    parser.add_argument('--exp_num', default='0.3', required=True)
    parser.add_argument('--nocs', default='part', help='which sub test set to choose')

    parser.add_argument('--hand', action='store_true', help='whether to visualize hand')
    parser.add_argument('--mano', action='store_true', help='whether to visualize hand')
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

    all_rts = {}
    all_bad             = []
    problem_ins = []
    start_time = time.time()
    #
    mano_layer_right = ManoLayer(
            mano_root=mano_path , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
    pca_hand = np.load(second_path + f'/data/pickle/hand_pca.npy',allow_pickle=True).item()
    mean_u  = pca_hand['M'] # 63
    eigen_v = pca_hand['E'] # 30*63
    error_all = []
    for i in range(len(test_group)):
        # try:
        h5_file    =  test_h5_path + '/' + test_group[i]
        print('')
        print('----------Now checking {}: {}'.format(i, h5_file))
        if test_group[i].split('_')[0] in problem_ins:
            print('skipping ', test_group[0][i])
            continue
        hf             =  h5py.File(h5_file, 'r')
        basename       =  hf.attrs['basename']

        nocs_gt        =  {}
        nocs_pred      =  {}
        nocs_gt['pn']  =  hf['nocs_per_point_gt'][()].transpose(1, 0)
        nocs_pred['pn']=  hf['nocs_per_point_pred'][()].transpose(1, 0)
        # nocs_pred['gn']=  hf['gocs_per_point_pred'][()].transpose(1, 0)

        input_pts      =  hf['P_gt'][()].transpose(1, 0)
        name_info      = basename.split('_')
        instance       = name_info[0]
        art_index      = name_info[1]
        grasp_ind      = name_info[2]
        frame_order    = name_info[3]
        mask_gt        =  hf['partcls_per_point_gt'][()]
        mask_pred      =  hf['partcls_per_point_pred'][()].transpose(1, 0)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> input, segmentation, nocs, poses
        # if args.viz:
        #     plot3d_pts([[input_pts]], [['']], s=2**2, title_name=['Input pts'], limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]], axis_off=True)
        part_idx_list_gt     = []
        part_idx_list_pred   = []
        cls_per_pt_pred= np.argmax(mask_pred, axis=1)
        nocs_err = []
        rt_gt    = []
        scale_gt = []

        for j in range(num_parts):
            part_idx_list_gt.append(np.where(mask_gt==j)[0])
            part_idx_list_pred.append(np.where(cls_per_pt_pred==j)[0])

            # if args.nocs == 'part':
            #     a = nocs_gt['pn'][part_idx_list_gt[j], :]
            #     b = nocs_pred['pn'][part_idx_list_gt[j], 3*j:3*(j+1)]
            # else:
            #     a = nocs_gt['gn'][part_idx_list_gt[j], :]
            #     if nocs_pred['gn'].shape[1] ==3:
            #         b = nocs_pred['gn'][part_idx_list_gt[j], :3]
            #     else:
            #         b = nocs_pred['gn'][part_idx_list_gt[j], 3*j:3*(j+1)]
            # c = input_pts[part_idx_list_gt[j], :]
            # nocs_err.append(np.mean(np.linalg.norm(a - b, axis=1)))
            # print('')
            # print(f'Part {j} GT nocs: \n', b)
            # print(f'Part {j} Pred nocs: \n', b)
            # # if args.viz:
            # #     plot3d_pts([[a, b]], [['GT part {}'.format(j), 'Pred part {}'.format(j)]], s=5**2, title_name=['NPCS comparison'], limits = [[0, 1], [0, 1], [0, 1]])
            # #     plot3d_pts([[a], [a]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=5**2, title_name=['GT NPCS', 'Pred NPCS'], color_channel=[[a], [b]], save_fig=True, limits = [[0, 1], [0, 1], [0, 1]], sub_name='{}'.format(j))
            # #     plot3d_pts([[a]], [['Part {}'.format(j)]], s=5**2, title_name=['NOCS Error'], color_channel=[[np.linalg.norm(a - b, axis=1)]], limits = [[0, 1], [0, 1], [0, 1]], colorbar=True)
            # #     plot3d_pts([[a], [c]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=5**2, title_name=['GT NOCS', 'point cloud'],limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], color_channel=[[a], [a]], save_fig=True, sub_name='{}'.format(j))
            # s, r, t, rt = estimateSimilarityUmeyama(a.transpose(), c.transpose())
            # rt_gt.append(compose_rt(r, t))
            # scale_gt.append(s)
            # # if args.viz and j==3:
            # #     d = np.matmul(a*s, r) + t
            # #     plot3d_pts([[d, c]], [['Part {} align'.format(j), 'Part {} cloud'.format(j)]], s=15, title_name=['estimation viz'], color_channel=[[a, a]], save_fig=True, sub_name='{}'.format(j))
        if args.viz:
            plot3d_pts([[input_pts[part_idx_list_gt[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=5**2, title_name=['GT seg'], sub_name=str(i),  axis_off=False, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)
            plot3d_pts([[input_pts[part_idx_list_pred[j], :] for j in range(num_parts)]], [['Part {}'.format(j) for j in range(num_parts)]], s=5**2, title_name=['Pred seg'], sub_name=str(i), axis_off=False, save_fig=args.save_fig, save_path=root_dset + '/NOCS/' + args.item)

        print('')
        #>>>>>>>>>>>>>>>>>>>>>>>>>> hands heatmap, unitvec
        if args.hand:
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
                if args.viz:
                    plot_arrows(input_pts[idx][ind_vote], [offset_gt[idx][ind_vote], offset_pred[idx][ind_vote]], whole_pts=input_pts, title_name='{}th_joint_voting'.format(j), limits = [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], dpi=200, s=25, thres_r=0.2, sparse=True, save=args.save_fig, index=i)

            if args.mano:
                # breakpoint()
                hand_joints_gt   = hf['handjoints_gt'][()].reshape(-1, 3)/100
                hand_joints_pred = hf['handjoints_pred'][()].reshape(-1, 3)/100
                hand_params_gt   = hf['regression_params_gt'][()].reshape(1, -1)
                hand_params_pred = hf['regression_params_pred'][()].reshape(1, -1)
                scale = 100
                verts = []
                faces = []
                for hand_params in [hand_params_gt, hand_params_pred]:
                    hand_vertices, hand_joints = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(hand_params))
                    hand_joints = hand_joints.cpu().data.numpy()[0]/scale

                    hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale
                    print(f'hand vertices have {hand_vertices.shape[0]} pts')
                    hand_faces = mano_layer_right.th_faces.cpu().data.numpy()
                    print(f'hand faces {hand_faces.shape} is {hand_faces[:3, :]}')
                    verts.append(hand_vertices)
                    faces.append(hand_faces)

                plot_hand_w_object(obj_verts=verts[0], obj_faces=faces[0], hand_verts=verts[1], hand_faces=faces[1],save=False)
                # hand_vertices_gt = hf['handvertices_gt'][()].reshape(-1, 3)/100
                # hand_vertices_pred = hf['handvertices_pred'][()].reshape(-1, 3)/100
            else:
                hand_joints_gt   = hf['regression_params_gt'][()].reshape(-1, 3)
                hand_joints_pred = hf['regression_params_pred'][()].reshape(-1, 3)
                print('')
                print('hand_joints_gt: \n', hand_joints_gt)
                print('hand_joints_pred: \n', hand_joints_pred)
                print('---hand_joints regression error: \n', np.linalg.norm(hand_joints_pred-hand_joints_gt, axis=1))
                # plot3d_pts([[input_pts, hand_joints_gt, hand_joints_pred]], [['Input pts', 'GT joints', 'Pred joints']], s=8**2, title_name=['Direct joint regression'], limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])

            results['regression'] = hand_joints_pred
            results['gt']         = hand_joints_gt

            for j in range(21):
                diver = results['divergence'][j]
                if diver > thres_var:
                    results['final'].append(hand_joints_pred[j:j+1,:])
                else:
                    results['final'].append(results['vote'][j])
            pred_hands   = np.concatenate(results['final']).reshape(-1, 3)
            # pred_hands   =  (np.dot( np.dot(eigen_v.T, eigen_v), (pred_hands.resahpe(-1)-mean_u).reshape(-1, 1)) + mean_u.reshape(-1, 1)).reshape(-1, 3)
            error_dist = pred_hands.reshape(-1, 3) - results['gt']
            error_dist[np.where(error_dist>1)[0], np.where(error_dist>1)[1]] = 0
            error_dist[np.where(error_dist<-1)[0], np.where(error_dist<-1)[1]] = 0
            error_all.append(error_dist) # J, 3
    error_norm = np.mean( np.linalg.norm(np.stack(error_all, axis=0), axis=2), axis=0)
    print('experiment: ', args.exp_num)
    for j, err in enumerate(error_norm):
        print(err)
    print('done!')
