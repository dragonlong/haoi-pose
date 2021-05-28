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
    def __init__(self, target_category='airplane', name_dset='modelnet40aligned', method_type='ours'):
        self.log_dir = 'default'
        self.symmetry_type = 0    # 0 for non-symmetric, 1 for symmetric;
        self.chosen_axis   = None
        self.name_dset     = name_dset
        self.target_category=target_category
        self.method_type=method_type
        if self.method_type=='ours':
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
                    self.exp_num    = '0.8562'
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
        elif self.method_type=='EPN':
            if self.name_dset == 'modelnet40aligned':
                if self.target_category == 'airplane':
                    self.exp_num    = '0.813a1'
                elif self.target_category == 'car':
                    self.exp_num    = '0.85a'
                elif self.target_category == 'chair':
                    self.exp_num    = '0.858a'
                elif self.target_category == 'sofa':
                    self.exp_num    = '0.859a'
                elif self.target_category == 'bottle':
                    self.exp_num    = '0.8562a'
            elif self.name_dset == 'modelnet40new':
                if self.target_category == 'airplane':
                    self.exp_num    = '0.91a'
                elif self.target_category == 'car':
                    self.exp_num    = '0.92a'
                elif self.target_category == 'chair':
                    self.exp_num    = '0.95a'
                elif self.target_category == 'sofa':
                    self.exp_num    = '0.96a'
                elif self.target_category == 'bottle':
                    self.exp_num    = '0.94a'
        elif self.method_type=='kpconv':
            if self.name_dset == 'modelnet40aligned':
                if self.target_category == 'airplane':
                    self.exp_num    = '0.813b'
                elif self.target_category == 'car':
                    self.exp_num    = '0.85b'
                elif self.target_category == 'chair':
                    self.exp_num    = '0.858b1'
                elif self.target_category == 'sofa':
                    self.exp_num    = '0.859b1'
                elif self.target_category == 'bottle':
                    self.exp_num    = '0.8562b1'
            elif self.name_dset == 'modelnet40new':
                if self.target_category == 'airplane':
                    self.exp_num    = '0.91b1'
                elif self.target_category == 'car':
                    self.exp_num    = '0.92b'
                elif self.target_category == 'chair':
                    self.exp_num    = '0.95b'
                elif self.target_category == 'sofa':
                    self.exp_num    = '0.96b'
                elif self.target_category == 'bottle':
                    self.exp_num    = '0.94b'
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
    for method_type in ['EPN', 'kpconv']:
        for name_dset in ['modelnet40new', 'modelnet40aligned']:
            for target_category in ['airplane', 'car', 'chair', 'sofa', 'bottle']:
                cfg = simple_config(name_dset=name_dset, target_category=target_category, method_type=method_type)
                print('going over ', name_dset, target_category, method_type)
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

#
# going over  modelnet40new airplane
# ---we have 6000 samples
# mid_r: 	 4.426689147949219
# mid_t: 	 0.09282216057181358
# 5deg5cm: 	 0.10016666666666667
#
# going over  modelnet40new car
# ---we have 6000 samples
# mid_r: 	 93.17675018310547
# mid_t: 	 0.26953746378421783
# 5deg5cm: 	 0.0015
#
# going over  modelnet40new chair
# ---we have 6000 samples
# mid_r: 	 3.2330715656280518
# mid_t: 	 0.08975344896316528
# 5deg5cm: 	 0.13183333333333333
#
# going over  modelnet40new sofa
# ---we have 6000 samples
# mid_r: 	 3.0072609186172485
# mid_t: 	 0.08405464142560959
# 5deg5cm: 	 0.147
#
# going over  modelnet40new bottle
# ---we have 6000 samples
# mid_r: 	 15.35927438735962
# mid_t: 	 0.11302925273776054
# 5deg5cm: 	 0.0315
#
# going over  modelnet40aligned airplane
# ---we have 100 samples
# mid_r: 	 1.120124638080597
# mid_t: 	 0.0
# 5deg5cm: 	 0.96
#
# going over  modelnet40aligned car
# ---we have 100 samples
# mid_r: 	 1.8522198796272278
# mid_t: 	 0.0
# 5deg5cm: 	 0.95
#
# going over  modelnet40aligned chair
# ---we have 100 samples
# mid_r: 	 3.8673394918441772
# mid_t: 	 0.0
# 5deg5cm: 	 0.68
#
# going over  modelnet40aligned sofa
# ---we have 100 samples
# mid_r: 	 1.5589593648910522
# mid_t: 	 0.0
# 5deg5cm: 	 0.97
#
# going over  modelnet40aligned bottle
# ---we have 100 samples
# mid_r: 	 42.506561279296875
# mid_t: 	 0.0
# 5deg5cm: 	 0.0
#
# going over  modelnet40new airplane
# ---we have 6000 samples
# mid_r: 	 103.47308731079102
# mid_t: 	 0.10249442607164383
# 5deg5cm: 	 0.0
#
# going over  modelnet40new car
# ---we have 6000 samples
# mid_r: 	 113.37484741210938
# mid_t: 	 0.1664542332291603
# 5deg5cm: 	 0.0
#
# going over  modelnet40new chair
# ---we have 6000 samples
# mid_r: 	 14.583336353302002
# mid_t: 	 0.07323120534420013
# 5deg5cm: 	 0.017333333333333333
#
# going over  modelnet40new sofa
# ---we have 6000 samples
# mid_r: 	 16.895160675048828
# mid_t: 	 0.06073477119207382
# 5deg5cm: 	 0.018
#
# going over  modelnet40new bottle
# ---we have 6000 samples
# mid_r: 	 1.9699054956436157
# mid_t: 	 0.08764144405722618
# 5deg5cm: 	 0.2265
#
# going over  modelnet40aligned airplane
# ---we have 100 samples
# mid_r: 	 10.780577659606934
# mid_t: 	 0.0
# 5deg5cm: 	 0.12
#
# going over  modelnet40aligned car
# ---we have 100 samples
# mid_r: 	 19.259392738342285
# mid_t: 	 0.0
# 5deg5cm: 	 0.06
#
# going over  modelnet40aligned chair
# ---we have 100 samples
# mid_r: 	 12.014530181884766
# mid_t: 	 0.0
# 5deg5cm: 	 0.06
#
# going over  modelnet40aligned sofa
# ---we have 100 samples
# mid_r: 	 141.0768814086914
# mid_t: 	 0.0
# 5deg5cm: 	 0.0
#
# going over  modelnet40aligned bottle
# ---we have 100 samples
# mid_r: 	 1.2128830552101135
# mid_t: 	 0.0
# 5deg5cm: 	 1.0
#
# (ptrans36) [lxiaol9@ca223 evaluation]$ python save_err_txt.py
# going over  modelnet40new airplane EPN
# ---we have 6000 samples
# mid_r: 	 4.426689147949219
# mid_t: 	 0.09282216057181358
# 5deg5cm: 	 0.10016666666666667
#
# going over  modelnet40new car EPN
# ---we have 6000 samples
# mid_r: 	 93.17675018310547
# mid_t: 	 0.26953746378421783
# 5deg5cm: 	 0.0015
#
# going over  modelnet40new chair EPN
# ---we have 6000 samples
# mid_r: 	 3.2330715656280518
# mid_t: 	 0.08975344896316528
# 5deg5cm: 	 0.13183333333333333
#
# going over  modelnet40new sofa EPN
# ---we have 6000 samples
# mid_r: 	 3.0072609186172485
# mid_t: 	 0.08405464142560959
# 5deg5cm: 	 0.147
#
# going over  modelnet40new bottle EPN
# ---we have 6000 samples
# mid_r: 	 15.35927438735962
# mid_t: 	 0.11302925273776054
# 5deg5cm: 	 0.0315
#
# going over  modelnet40aligned airplane EPN
# ---we have 100 samples
# mid_r: 	 1.120124638080597
# mid_t: 	 0.0
# 5deg5cm: 	 0.96
#
# going over  modelnet40aligned car EPN
# ---we have 100 samples
# mid_r: 	 1.8522198796272278
# mid_t: 	 0.0
# 5deg5cm: 	 0.95
#
# going over  modelnet40aligned chair EPN
# ---we have 100 samples
# mid_r: 	 3.8673394918441772
# mid_t: 	 0.0
# 5deg5cm: 	 0.68
#
# going over  modelnet40aligned sofa EPN
# ---we have 100 samples
# mid_r: 	 1.5589593648910522
# mid_t: 	 0.0
# 5deg5cm: 	 0.97
#
# going over  modelnet40aligned bottle EPN
# ---we have 100 samples
# mid_r: 	 42.506561279296875
# mid_t: 	 0.0
# 5deg5cm: 	 0.0
#
# going over  modelnet40new airplane kpconv
# ---we have 6000 samples
# mid_r: 	 103.47308731079102
# mid_t: 	 0.10249442607164383
# 5deg5cm: 	 0.0
#
# going over  modelnet40new car kpconv
# ---we have 6000 samples
# mid_r: 	 113.37484741210938
# mid_t: 	 0.1664542332291603
# 5deg5cm: 	 0.0
#
# going over  modelnet40new chair kpconv
# ---we have 6000 samples
# mid_r: 	 14.583336353302002
# mid_t: 	 0.07323120534420013
# 5deg5cm: 	 0.017333333333333333
#
# going over  modelnet40new sofa kpconv
# ---we have 6000 samples
# mid_r: 	 16.895160675048828
# mid_t: 	 0.06073477119207382
# 5deg5cm: 	 0.018
#
# going over  modelnet40new bottle kpconv
# ---we have 6000 samples
# mid_r: 	 1.9699054956436157
# mid_t: 	 0.08764144405722618
# 5deg5cm: 	 0.2265
#
# going over  modelnet40aligned airplane kpconv
# ---we have 100 samples
# mid_r: 	 10.780577659606934
# mid_t: 	 0.0
# 5deg5cm: 	 0.12
#
# going over  modelnet40aligned car kpconv
# ---we have 100 samples
# mid_r: 	 19.259392738342285
# mid_t: 	 0.0
# 5deg5cm: 	 0.06
#
# going over  modelnet40aligned chair kpconv
# ---we have 100 samples
# mid_r: 	 12.014530181884766
# mid_t: 	 0.0
# 5deg5cm: 	 0.06
#
# going over  modelnet40aligned sofa kpconv
# ---we have 100 samples
# mid_r: 	 141.0768814086914
# mid_t: 	 0.0
# 5deg5cm: 	 0.0
#
# going over  modelnet40aligned bottle kpconv
# ---we have 100 samples
# mid_r: 	 1.2128830552101135
# mid_t: 	 0.0
# 5deg5cm: 	 1.0
#
# (ptrans36) [lxiaol9@ca223 evaluation]$
