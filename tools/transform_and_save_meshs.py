import numpy as np
import sys
import os
import argparse
import h5py
import yaml
import argparse
import subprocess
import xml.etree.ElementTree as ET
from collections import OrderedDict

import _init_paths
from global_info import global_info
from common.d3_utils import point_rotate_about_axis
from common.data_utils import get_urdf, load_model_split, save_objmesh
from common.transformations import quaternion_about_axis, quaternion_matrix

def breakpoint():
    import pdb;pdb.set_trace()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--cnt', default=10, help='count of articulation change')
    parser.add_argument('--save_path', default='/home/dragon/Documents/ICML2021/data/objs/', help='save uniform mesh')
    args = parser.parse_args()

    infos           = global_info()
    my_dir          = infos.base_path
    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    name_dset       = dset_info.dataset_name
    unseen_instances= dset_info.test_list
    test_ins        = dset_info.test_list
    special_ins     = dset_info.spec_list

    #>>>>>>>>>>>>>>>>>>>>>>>>>> internal config >>>>>>>>>>>>>>>>>>>>>>#
    data_root     = my_dir + '/dataset/' + name_dset
    all_ins       = os.listdir(data_root + '/urdf/' + args.item)
    all_ins.sort()
    save_path     = args.save_path + '/' + args.item
    path_urdf     = data_root + '/urdf/' + args.item

    angles = np.random.rand(len(all_ins), args.cnt, num_parts-1)
    np.save(save_path + f'/{args.item}.npy', angles)

    # for i in range(1):
    for i in range(len(all_ins)):
        cur_urdf   = all_ins[i]
        tree_urdf = ET.parse("{}/{}/syn.urdf".format(path_urdf, cur_urdf))
        root      = tree_urdf.getroot()

        num_joints = 0
        num_joints = len(os.listdir("{}/{}/".format(path_urdf, cur_urdf))) -2
        urdf_ins   = get_urdf("{}/{}".format(path_urdf, cur_urdf))
        num_joints = len(urdf_ins['obj_name']) -1

        # angles = np.linspace(0, 1, args.cnt)
        prefix    = ['mtllib none_motion.mtl', 'usemtl material_5131']
        directory = '{}/{}/'.format(save_path, cur_urdf) # 
        if not os.path.exists( directory ):
            os.makedirs(directory)
        for p in range(args.cnt):
            save_name = '{}/{}/{}.obj'.format(save_path, cur_urdf, p)
            whole_obj_dict = {'v':[], 'n':[], 't':[], 'f':[]}
            ind_link = 0
            start_vind = 0
            name_list = []
            for k, obj_files in enumerate(urdf_ins['obj_name']):
                num_vertices = 0
                if obj_files is not None and not isinstance(obj_files, list):
                    dict_mesh, _, _, _ = load_model_split(obj_files)
                    name_obj  = obj_files.split('.')[0].split('/')[-1]
                    name_list.append(name_obj) 
                    anchor = np.array(urdf_ins['joint']['xyz'][k])
                    univec = np.array(urdf_ins['joint']['axis'][k])
                    theta  = angles[i, p, ind_link] * np.pi/2
                    for j in range(len(dict_mesh['v'])):
                        if name_obj !='none_motion':
                            dict_mesh['v'][j] = point_rotate_about_axis(np.array(dict_mesh['v'][j]), anchor, univec, theta)
                            dict_mesh['n'][j] = np.dot(np.array(dict_mesh['n'][j]), quaternion_matrix(quaternion_about_axis(theta, univec))[:3, :3].T)
                        whole_obj_dict['v'].append(dict_mesh['v'][j])
                        whole_obj_dict['n'].append(dict_mesh['n'][j])
                        num_vertices += len(dict_mesh['v'][j])
                        if start_vind == 0:
                            whole_obj_dict['f'].append(dict_mesh['f'][j])
                        else:
                            for m in range(len(dict_mesh['f'][j])):
                                for n in range(len(dict_mesh['f'][j][m])):
                                    inds = dict_mesh['f'][j][m][n].split('/')
                                    dict_mesh['f'][j][m][n] = f'{int(inds[0])+int(start_vind)}/{int(inds[0])+int(start_vind)}/{int(inds[0])+int(start_vind)}'
                            whole_obj_dict['f'].append(dict_mesh['f'][j])
                    if name_obj !='none_motion':
                        ind_link+=1
                    start_vind += num_vertices

            print(cur_urdf, name_list, angles)
            save_objmesh(save_name, whole_obj_dict, prefix=prefix)
        subprocess.run(["cp", f'{data_root}/objects/{args.item}/0001/part_objs/none_motion.mtl', f'{save_path}/{cur_urdf}/'])

