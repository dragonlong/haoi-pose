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
from lib.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list
from lib.eval_utils import hungarian_matching
from lib.data_utils import get_demo_h5, get_full_test
from lib.aligning import estimateSimilarityTransform, estimateSimilarityUmeyama
from global_info import global_info

def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation.transpose()
    aligned_RT[:3, 3]  = translation
    aligned_RT[3, 3]   = 1
    return aligned_RT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='unseen', help='which sub test set to choose')
    parser.add_argument('--nocs', default='part', help='which sub test set to choose')
    parser.add_argument('--item', default='oven', help='object category for benchmarking')
    parser.add_argument('--save', action='store_true', help='save err to pickles')
    parser.add_argument('--viz', action='store_true', help='whether to viz')
    args = parser.parse_args()

    infos           = global_info()


    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    unseen_instances= dset_info.test_list
    special_ins     = dset_info.spec_list
    main_exp        = dset_info.exp
    baseline_exp    = dset_info.baseline

    if args.domain == 'demo':
        base_path       = my_dir + '/results/demo'
        test_h5_path    = base_path + '/{}'.format(baseline_exp)
        test_group      = get_demo_h5(os.listdir(test_h5_path))
    else:
        base_path       = my_dir + '/results/test_pred' # testing
        test_h5_path    = base_path + '/{}'.format(main_exp)
        test_group      = get_full_test(os.listdir(test_h5_path), unseen_instances, domain=args.domain, spec_instances=special_ins)
    
    all_bad          = []
    print('we have {} testing data for {} {}'.format(len(test_group), args.domain, args.item))
    all_rts = {}
    file_name = base_path + '/pickle/{}/{}_{}_{}_rt.pkl'.format(main_exp, args.domain, args.nocs, args.item)
    save_path = base_path + '/pickle/{}/'.format(main_exp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    problem_ins = []
    start_time = time.time()
    for i in range(len(test_group)):
        try:
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
            frame_order    = name_info[2]

            mask_pred      =  hf['instance_per_point'][()]

            part_idx_list_gt   = []
            part_idx_list_pred   = []
            cls_per_pt_pred= np.argmax(mask_pred, axis=1)
            nocs_err = []
            rt_gt    = []
            scale_gt = []
            for j in range(num_parts):
                part_idx_list_gt.append(np.where(mask_gt==j)[0])
                part_idx_list_pred.append(np.where(cls_per_pt_pred==j)[0])

                if args.viz:
                    plot3d_pts([[gt_pn[j], pred_pn[j]]], [['GT part {}'.format(j), 'Pred part {}'.format(j)]], s=15, title_name=['part NOCS comparison'])
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
                # plot3d_pts([[a], [b]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=15, title_name=['GT NOCS', 'Pred NOCS'], color_channel=[[a], [a]], save_fig=True, sub_name='{}'.format(j))
                # plot3d_pts([[a]], [['Part {}'.format(j)]], s=15, title_name=['NOCS Error'], color_channel=[[np.linalg.norm(a - b, axis=1)]],  colorbar=True)
                # plot3d_pts([[a], [c]], [['Part {}'.format(j)], ['Part {}'.format(j)]], s=15, title_name=['GT NOCS', 'point cloud'], color_channel=[[a], [a]], save_fig=True, sub_name='{}'.format(j))
                s, r, t, rt = estimateSimilarityUmeyama(a.transpose(), c.transpose())
                rt_gt.append(compose_rt(r, t))
                scale_gt.append(s)
                if args.viz:
                    d = np.matmul(a*s, r) + t
                    plot3d_pts([[d, c]], [['Part {} align'.format(j), 'Part {} cloud'.format(j)]], s=15, title_name=['estimation viz'], color_channel=[[a, a]], save_fig=True, sub_name='{}'.format(j))

            rts_dict   = {}
            scale_dict = {}
            rt_dict    = {}
            rt_dict['gt']    = rt_gt
            scale_dict['gt'] = scale_gt
            rts_dict['scale']  = scale_dict
            rts_dict['rt']     = rt_dict
            rts_dict['nocs_err'] = nocs_err
            all_rts[basename]  = rts_dict
        except:
            # print('skipping ', test_group[i])
            pass

    if args.save:
        with open(file_name, 'wb') as f:
            pickle.dump(all_rts, f, protocol=2)
            print('saving to ', file_name)
    # print('takes {} seconds', time.time() - start_time)

    # plot3d_pts([[input_pts[part_idx_list_gt[0], :], input_pts[part_idx_list_gt[1], :], input_pts[part_idx_list_gt[2], :]]], [['Part {}'.format(j) for j in range(3)]], s=5, title_name=['GT Seg'])
    # plot3d_pts([[input_pts[part_idx_list_pred[0], :], input_pts[part_idx_list_pred[1], :], input_pts[part_idx_list_pred[2], :]]], [['Part {}'.format(j) for j in range(3)]], s=5, title_name=['Pred Seg'])
    # plot3d_pts([[nocs_gt[part_idx_list_pred[0], :], nocs_gt[part_idx_list_pred[1], :], nocs_gt[part_idx_list_pred[2], :]]], [['Part {}'.format(j) for j in range(3)]], s=5, title_name=['GT NOCS'])
    # plot3d_pts([[nocs_pred[part_idx_list_pred[0], 3:6], nocs_pred[part_idx_list_pred[1], 0:3], nocs_pred[part_idx_list_pred[2], 6:9]]], [['Part {}'.format(j) for j in range(3)]], s=5, title_name=['Pred NOCS'])
    # plot3d_pts([[input_pts]], [['']], s=5, title_name=['NOCS Confidence'], color_channel=[[np.concatenate([confidence, np.zeros((confidence.shape[0], 2))], axis=1)]])
