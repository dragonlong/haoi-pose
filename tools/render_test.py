"""
Func: data rendering from URDF
      it will render data with:
    - rgb image;
    - depth image;
    - part masks;
    - pose labels;
    - joint states;
# make sure to compile pybullet with PYBULLET_USE_NUMPY enabled
"""
import numpy as np
import pybullet

# here we add one
import sys
import os
import time
import cv2
import h5py
import yaml
import argparse
import threading
import subprocess
import xml.etree.ElementTree as ET
from collections import OrderedDict
# try to use LibYAML bindings if possible
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from yaml.representer import SafeRepresenter
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())
def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))

Dumper.add_representer(OrderedDict, dict_representer)
Loader.add_constructor(_mapping_tag, dict_constructor)
Dumper.add_representer(str, SafeRepresenter.represent_str)

# custom libs
import _init_paths
from global_info import global_info
from lib.data_utils import get_model_pts, get_urdf, load_model_split, save_objmesh
from lib.d3_utils import point_rotate_about_axis
# Step through simulation time
def step_simulation():
    # self._sim_time = 0
    while True:
        pybullet.stepSimulation()
        # self._sim_time += self._sim_timestep
        time.sleep(0.01)

def breakpoint():
    import pdb;pdb.set_trace()

#>>>>>>>>>>>>>>>>>>>>>>---------Rendering setup----------<<<<<<<<<<<<<<<<<<<<<<<<<#
def render_data(data_root, name_obj, cur_urdf, args=None, cam_dis=1, urdf_file='NA', _WRITE_FLAG=True, _RENDER_FLAG=True, _CREATE_FOLDER=True, RENDER_NUM=100, ARTIC_CNT=20, _RENDER_MODE='random', _USE_GUI=True, _IS_DUBUG=True):
    #>>>>>>>>>>>>>>>>>>>>>>>>>> internal config >>>>>>>>>>>>>>>>>>>>>>#
    save_path     = data_root + '/tf_objects/' + name_obj
    path_urdf     = data_root + '/urdf/' + name_obj

    # parse urdf
    tree_urdf = ET.parse("{}/{}/syn.urdf".format(path_urdf, cur_urdf))
    root      = tree_urdf.getroot()

    num_joints = 0
    num_joints = len(os.listdir("{}/{}/".format(path_urdf, cur_urdf))) -2
    urdf_ins   = get_urdf("{}/{}".format(path_urdf, cur_urdf))
    num_joints = len(urdf_ins['obj_name']) -1

    # instance-wise offset for camera distance
    name_list = []
    # angles = np.random.rand(num_joints + 1)
    angles = [-0.5, -0.5]
    for k, obj_files in enumerate(urdf_ins['obj_name']):
        prefix = None
        if obj_files is not None and not isinstance(obj_files, list):
            print('now processing ', obj_files)
            dict_mesh, _, _, _ = load_model_split(obj_files)
            name_obj  = obj_files.split('.')[0].split('/')[-1]
            name_list.append(name_obj) # which should follow the right order
            # save raw objs
            # save obj files to video folder
            directory = f'{save_path}/{cur_urdf}/'
            if not os.path.exists( directory ):
                os.makedirs(directory)
            save_name = f'{save_path}/{cur_urdf}/original_{name_obj}.obj'
            if k == 0:
                prefix = ['mtllib none_motion.mtl', 'usemtl material_5131']
            save_objmesh(save_name, dict_mesh, prefix=prefix)
            # breakpoint()
            # rotate model pts around joint
            if name_obj !='none_motion':
                anchor = np.array(urdf_ins['joint']['xyz'][k])
                univec = np.array(urdf_ins['joint']['axis'][k])
            else:
                anchor = np.array(urdf_ins['joint']['xyz'][k+1])
                univec = np.array(urdf_ins['joint']['axis'][k+1])
            theta  = angles[k-1]*np.pi/2
            for j in range(len(dict_mesh['v'])):
                dict_mesh['v'][j] = point_rotate_about_axis(np.array(dict_mesh['v'][j]), anchor, univec, theta)
            # save obj files to video folder
            directory = f'{save_path}/{cur_urdf}/'
            if not os.path.exists( directory ):
                os.makedirs(directory)
            save_name = f'{save_path}/{cur_urdf}/new_{name_obj}.obj'
            if k == 0:
                prefix = ['mtllib none_motion.mtl', 'usemtl material_5131']
            save_objmesh(save_name, dict_mesh, prefix=prefix)
    subprocess.run(["cp", f'{data_root}/objects/eyeglasses/0001/part_objs/none_motion.mtl', f'{save_path}/{cur_urdf}/'])

if __name__ == "__main__":
    #>>>>>>>>>>>>>>>>>>>>>>>>>> config regions >>>>>>>>>>>>>>>>>>>>>>>>#
    infos     = global_info()
    my_dir    = infos.base_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='indicating whether in debug mode')
    parser.add_argument('--dataset', default='shape2motion', help='name of the dataset we use')
    parser.add_argument('--item', default='laptop', help='name of category we use')
    parser.add_argument('--dis',   default=3, help='default camera2object distance')
    parser.add_argument('--mode',  default='train', help='mode decides saving folder:train/demo')
    parser.add_argument('--roll', default='30,40', help='camera view angle')
    parser.add_argument('--pitch', default='30,40', help='camera view angle')
    parser.add_argument('--yaw',  default='30,40', help='camera view angle')
    parser.add_argument('--min_angles',  default='30,30', help='minimum joint angles')
    parser.add_argument('--max_angles',  default='90,90', help='maximum joint angles')
    parser.add_argument('--cnt', default=30, help='count of articulation change')
    parser.add_argument('--num', default=10, help='number of rendering per articulation')
    args = parser.parse_args()
    #>>>>>>>>>>>>>>>>>>>>>>>> config end here >>>>>>>>>>>>>>>>>>>>>>>>>#

    is_debug = args.debug
    if is_debug:
        _WRITE   = False
        _RENDER  = True
        _CREATE  = True
        _USE_GUI = True
    else:
        _WRITE   = True
        _RENDER  = True
        _CREATE  = True
        _USE_GUI = False

    num_render     = int(args.num)                                               # viewing angles
    cnt_artic      = int(args.cnt)
    cam_dis        = float(args.dis)                                             # articulation change
    name_dataset   = args.dataset

    data_root = my_dir + '/dataset/' + name_dataset
    all_ins   = os.listdir(data_root + '/urdf/' + args.item)
    all_ins.sort()

    np.random.seed(5) # better to have this random seed here
    if is_debug:
        for instance in all_ins:
            render_data(data_root, args.item, instance, cam_dis=cam_dis, args=args,  _WRITE_FLAG=_WRITE, _RENDER_FLAG=_RENDER, _CREATE_FOLDER=_CREATE, RENDER_NUM=num_render, ARTIC_CNT=cnt_artic, _USE_GUI=_USE_GUI, _IS_DUBUG=is_debug)
    else:
        for instance in all_ins: #todo
            render_data(data_root, args.item, instance, cam_dis=cam_dis, args=args,  _WRITE_FLAG=_WRITE, _RENDER_FLAG=_RENDER, _CREATE_FOLDER=_CREATE, RENDER_NUM=num_render, ARTIC_CNT=cnt_artic, _USE_GUI=_USE_GUI, _IS_DUBUG=is_debug)
