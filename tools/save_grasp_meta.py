import os
import argparse
import glob
import cv2
from PIL import Image
import trimesh
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
import json
from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments

from sklearn.decomposition import PCA
from manopth import rodrigues_layer
from manopth.manolayer import ManoLayer

import _init_paths
from global_info import global_info
from common.data_utils import save_objmesh, fast_load_obj
from common.vis_utils import plot_hand_w_object

def breakpoint():
    import pdb;pdb.set_trace()

def print_collision_result(o1_name, o2_name, result):
    print('Collision between {} and {}:'.format(o1_name, o2_name))
    print('-'*30)
    print('Collision?: {}'.format(result.is_collision))
    print('Number of contacts: {}'.format(len(result.contacts)))
    print('')

def print_continuous_collision_result(o1_name, o2_name, result):
    print('Continuous collision between {} and {}:'.format(o1_name, o2_name))
    print('-'*30)
    print('Collision?: {}'.format(result.is_collide))
    print('Time of collision: {}'.format(result.time_of_contact))
    print('')

def print_distance_result(o1_name, o2_name, result):
    print('Distance between {} and {}:'.format(o1_name, o2_name))
    print('-'*30)
    print('Distance: {}'.format(result.min_distance))
    print('Closest Points:')
    print(result.nearest_points[0])
    print(result.nearest_points[1])
    print('')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--item', default='eyeglasses', help='object category for benchmarking')
    parser.add_argument('--viz', action='store_true', help='whether to visualize')
    parser.add_argument('--save', action='store_true', help='whether to save')
    parser.add_argument('--viz_j', action='store_true', help='whether to visualize joints')
    parser.add_argument('--viz_c', action='store_true', help='whether to visualize contacts')
    parser.add_argument('--viz_n', action='store_true', help='whether to visualize normals')
    parser.add_argument('--viz_m', action='store_true', help='whether to visualize meshs')
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

    # mano_layer_right = ManoLayer(
    #         mano_root=mano_path , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
    mano_layer_right = ManoLayer(
            mano_root=mano_path , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    mano_data = mano_path +'/MANO_RIGHT.pkl'
    smpl_data = ready_arguments(mano_data)
    components = smpl_data['hands_components']

    # load glass pose in json
    # json_names = glob.glob( grasps_meta + f'/{args.item}/*json') # eyeglasses_0002_0_scale_200.json
    json_names = glob.glob( grasps_meta + f'/{args.item}/*0001_7*json') # eyeglasses_0002_0_scale_200.json
    json_names.sort()
    offset_contacts = {}
    joints_all   = {}
    contacts_all = {}
    vertices_all = {}
    poses_all    = {}
    trans_all    = {}

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

        # objname = f'{whole_obj}/{category}/{instance}/{arti_ind}.obj'
        # obj= fast_load_obj(open(objname, 'rb'))[0] # why it is [0]
        # obj_verts = obj['vertices']
        # obj_faces = obj['faces']
        obj_verts = None
        obj_faces = None
        for j in range(len(hand_attrs['grasps'])):
            print('loading ', j)
            save_name = save_dir + f'/{j}.obj'
            contacts = hand_attrs['grasps'][j]['contacts']
            jts = None
            nmls = None

            # contact_links = {}
            contact_pts = []
            contact_link  = []
            for contact in contacts:
                contact_pts.append(contact['pose'][:3])
                contact_link.append(contact['link'])
            contact_pts = np.array(contact_pts) * 1000/scale

            posesnew = np.array(hand_attrs['grasps'][j]['mano_pose']).reshape(1, -1)
            pca_pcs  = np.dot(posesnew[0, 3:].reshape(1, 45),  components) # 1* 45

            pca_input= np.concatenate([posesnew[:, :3], pca_pcs], axis=1)
            #
            mano_trans = hand_attrs['grasps'][j]['mano_trans']

            hand_vertices, hand_joints = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(posesnew), th_trans=torch.FloatTensor(mano_trans))
            # hand_vertices, hand_joints = mano_layer_right(torch.FloatTensor(pca_input), th_trans=torch.FloatTensor(mano_trans))
            hand_joints = hand_joints.cpu().data.numpy()[0]/scale

            hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale
            print(f'hand vertices have {hand_vertices.shape[0]} pts')
            hand_faces = mano_layer_right.th_faces.cpu().data.numpy()
            print(f'hand faces {hand_faces.shape} is {hand_faces[:3, :]}')

            if args.viz:
                plot_hand_w_object(obj_verts, obj_faces, hand_vertices, hand_faces, pts=contact_pts, jts=hand_joints, nmls=nmls, save_path=viz_dir + f'/{j}.png', viz_m=args.viz_m, viz_c=args.viz_c, viz_j=args.viz_j, viz_n=args.viz_n, save=True)
            joints_all[f'{instance}_{arti_ind}_{j}']   = hand_joints
            contacts_all[f'{instance}_{arti_ind}_{j}'] = contact_pts
            vertices_all[f'{instance}_{arti_ind}_{j}'] = hand_vertices
            poses_all[f'{instance}_{arti_ind}_{j}']    = posesnew
            trans_all[f'{instance}_{arti_ind}_{j}']    = np.array(mano_trans)

    if args.save:
        np.save(f'{grasps_meta}/{args.item}_joints.npy', joints_all)
        np.save(f'{grasps_meta}/{args.item}_contacts.npy', contacts_all)
        np.save(f'{grasps_meta}/{args.item}_vertices.npy', vertices_all)
        np.save(f'{grasps_meta}/{args.item}_poses.npy', poses_all)
        np.save(f'{grasps_meta}/{args.item}_trans.npy', trans_all)

    # print(offset_contacts)
    # mean_value = []
    # max_value  = []
    # keys = []
    # for key, value in offset_contacts.items():
    #     keys.append(key)
    #     mean_value.append(np.mean(np.array(value)))
    #     # breakpoint()
    #     max_value.append(max(value))
    #     print(key, mean_value[-1], max_value[-1])
    # print('keys: ', keys)
    # print('mean_value: ', mean_value)
    # print('max_value: ', max_value)

if __name__ == '__main__':
    main()
