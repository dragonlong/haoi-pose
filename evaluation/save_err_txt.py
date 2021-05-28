import os
import time
from os import makedirs, remove
from os.path import exists, join
from time import time
import numpy as np
import copy
from glob import glob
import scipy.io as sio
import argparse

import __init__
from global_info import global_info

infos     = global_info()
my_dir    = infos.base_path
second_path = infos.second_path
render_path = infos.render_path

def bp():
    import pdb;pdb.set_trace()

class simple_config(object):
    def __init__(self, target_category='airplane', name_dset='modelnet40aligned'):
        self.log_dir = 'default'
        self.symmetry_type = 0    # 0 for non-symmetric, 1 for symmetric;
        self.chosen_axis   = None
        self.name_dset     = name_dset
        self.target_category=target_category
        if self.name_dset == 'modelnet40aligned':
            if self.target_category == 'airplane':
                self.exp_num    = '0.813'
            elif self.target_category == 'car':
                self.exp_num    = '0.851'
            elif self.target_category == 'chair':
                self.exp_num    = '0.8581'
            elif self.target_category == 'sofa':
                self.exp_num    = '0.8591'
            elif self.target_category == 'bottle':
                self.exp_num    = '0.8561'
        elif self.name_dset == 'modelnet40new':
            if self.target_category == 'airplane':
                self.exp_num    = '0.913r'
            elif self.target_category == 'car':
                self.exp_num    = '0.92r'
            elif self.target_category == 'chair':
                self.exp_num    = '0.951r'
            elif self.target_category == 'sofa':
                self.exp_num    = '0.961r'
            elif self.target_category == 'bottle':
                self.exp_num    = '0.941r'

        self.dataset_path=f'{my_dir}/data/modelnet40aligned/EvenAlignedModelNet40PC'

def load_input(cfg):
    res_path = f'{my_dir}/results/test_pred/{cfg.name_dset}/{cfg.exp_num}_unseen_part_rt_pn_general.npy'
    results = np.load(res_path, allow_pickle=True).item()
    infos_dict, track_dict = results['info'], results['err']
    return infos_dict, track_dict

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name_dset', default='modelnet40aligned', help='object category for benchmarking')
    # parser.add_argument('--target_category', default='airplane', help='exact object category')
    # args = parser.parse_args()
    #
    # cfg = simple_config(name_dset=args.name_dset, target_category=args.target_category)
    for name_dset in ['modelnet40new', 'modelnet40aligned']:
        for target_category in ['airplane', 'car', 'chair', 'sofa', 'bottle']:
            cfg = simple_config(name_dset=name_dset, target_category=target_category)
            print('going over ', name_dset, target_category)
            # load npy '/model/${item}/${exp_num}'
            infos_dict, track_dict = load_input(cfg)
            basenames = infos_dict['basename']
            inputs   = np.array(infos_dict['in'])
            r_gt     = np.array(infos_dict['r_gt'])
            t_gt     = np.array(infos_dict['t_gt'])
            r_pred   = np.array(infos_dict['r_pred'])
            rdiff    = np.array(track_dict['rdiff'])
            tdiff    = np.array(track_dict['tdiff'])
            num      = len(basenames)
            print(f'---we have {num} samples')

            save_arr = np.stack([rdiff.reshape(-1), tdiff.reshape(-1)], axis=-1)
            print('mid_r: \t', np.median(rdiff))
            print('mid_t: \t', np.median(tdiff))
            tdiff_1 = tdiff[rdiff<=5.0]
            tdiff_2 = tdiff_1[tdiff_1<=0.05]
            print('5deg5cm: \t', len(tdiff_2)/len(tdiff))
            print('')
            # if cfg.name_dset == 'modelnet40aligned':
            #     fname =  f'{my_dir}/results/test_pred/rt_err/{cfg.name_dset}_{cfg.target_category}_unseen_complete_ours.txt'
            # else:
            #     fname =  f'{my_dir}/results/test_pred/rt_err/{cfg.name_dset}_{cfg.target_category}_unseen_partial_ours.txt'
            # np.savetxt(fname, save_arr, fmt='%1.5f')
            # print("save to txt with r err, t err", fname)
