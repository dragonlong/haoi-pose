import os
import argparse
import glob
import cv2
from PIL import Image
import openmesh as om
import trimesh
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
import json

# from sklearn.decomposition import PCA
from manopth import rodrigues_layer
from manopth.tensutils import th_posemap_axisang, make_list, th_pack, subtract_flat_id, th_with_zeros
from manopth.manolayer import ManoLayer

import _init_paths
from global_info import global_info
from common.data_utils import get_urdf, load_model_split, save_objmesh, fast_load_obj
from common.debugger import breakpoint, print_group
from common.vis_utils import plot_hand_w_object

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--viz', action='store_true', help='whether to visualize')
    parser.add_argument('--save', action='store_true', help='whether to visualize')
    args = parser.parse_args()

    infos           = global_info()
    my_dir          = infos.base_path
    dset_info       = infos.datasets[args.item]
    num_parts       = dset_info.num_parts
    num_ins         = dset_info.num_object
    name_dset       = dset_info.dataset_name
    grasps_meta     = infos.grasps_meta
    mano_path       = infos.mano_path
    whole_obj       = infos.whole_obj
    hand_mesh       = infos.hand_mesh
    viz_path        = infos.viz_path

    mano_layer_right = ManoLayer(
            mano_root=mano_path , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    # load glass pose in json
    # json_names = glob.glob( grasps_meta + '/*json') # eyeglasses_0002_0_scale_200.json
    json_names = glob.glob( grasps_meta + f'/{args.item}/*0001_7*json')
    json_names.sort()
    for json_name in json_names:
        name_attrs = json_name.split('.js')[0].split('/')[-1].split('_')
        category = name_attrs[0]
        instance = name_attrs[1]
        arti_ind = name_attrs[2]
        scale    = int(name_attrs[-1])
        save_dir = f'{hand_mesh}/{category}/{instance}/{arti_ind}'
        viz_dir  = f'{viz_path}/{category}/{instance}/{arti_ind}'
        if not os.path.exists( save_dir ):
            os.makedirs(save_dir)
        if not os.path.exists( viz_dir ):
            os.makedirs(viz_dir)
        with open(json_name) as json_file:
            hand_attrs = json.load(json_file)

        if args.viz:
            objname = f'{whole_obj}/{category}/{instance}/{arti_ind}.obj'
            obj= fast_load_obj(open(objname, 'rb'))[0] # why it is [0]
            obj_verts = obj['vertices']
            obj_faces = obj['faces']

        for j in range(len(hand_attrs['grasps'])):
            save_name = save_dir + f'/{j}.obj'
            # posesnew = np.concatenate(([hand['pca_manorot']], hand['pca_poses']), 1)
            posesnew   = np.array(hand_attrs['grasps'][j]['mano_pose']).reshape(1, -1)
            mano_trans = hand_attrs['grasps'][j]['mano_trans']
            print_group([posesnew, mano_trans], ['mano_poses', 'mano_trans'])
            hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(posesnew), th_trans=torch.FloatTensor(mano_trans))
            hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale
            print(f'hand vertices have {hand_vertices.shape[0]} pts')
            hand_faces = mano_layer_right.th_faces.cpu().data.numpy()
            print(f'hand faces {hand_faces.shape} is {hand_faces[:3, :]}')

            if args.save:
                mesh = trimesh.Trimesh(vertices=hand_vertices,
                           faces=hand_faces)
                mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
                with open(save_name,"w+") as fp:
                    fp.write(mesh_txt)

            if args.viz:
                plot_hand_w_object(obj_verts, obj_faces, hand_vertices, hand_faces, save_path=viz_dir + f'/{j}.png', save=True)

if __name__ == '__main__':
    main()
