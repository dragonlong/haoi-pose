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
import fcl
import igl

# from sklearn.decomposition import PCA
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

    mano_layer_right = ManoLayer(
            mano_root=mano_path , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    # load glass pose in json
    json_names = glob.glob( grasps_meta + f'/{args.item}/*json') # eyeglasses_0002_0_scale_200.json
    json_names.sort()
    offset_contacts = {}
    joints_all   = {}
    contacts_all = {}
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

        objname = f'{whole_obj}/{category}/{instance}/{arti_ind}.obj'
        obj= fast_load_obj(open(objname, 'rb'))[0] # why it is [0]
        obj_verts = obj['vertices']
        obj_faces = obj['faces']
        for j in range(len(hand_attrs['grasps'])):
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
            mano_trans = hand_attrs['grasps'][j]['mano_trans']
            hand_vertices, hand_joints = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(posesnew), th_trans=torch.FloatTensor(mano_trans))
            hand_joints = hand_joints.cpu().data.numpy()[0]/scale

            hand_vertices = hand_vertices.cpu().data.numpy()[0]/scale
            print(f'hand vertices have {hand_vertices.shape[0]} pts')
            hand_faces = mano_layer_right.th_faces.cpu().data.numpy()
            print(f'hand faces {hand_faces.shape} is {hand_faces[:3, :]}')

            # # # Create mesh geometry
            # # mesh_hand = fcl.BVHModel()
            # # mesh_hand.beginModel(len(hand_vertices), len(hand_faces))
            # # mesh_hand.addSubModel(hand_vertices, hand_faces)
            # # mesh_hand.endModel()
            #
            # # req = fcl.CollisionRequest(num_max_contacts=5, enable_contact=True)
            # # res = fcl.CollisionResult()
            #
            # # n_contacts = fcl.collide(fcl.CollisionObject(mesh, fcl.Transform()),
            # #                          fcl.CollisionObject(mesh_hand, fcl.Transform()),
            # #                          req, res)
            # # print('n_contacts: ', n_contacts)
            # # print_collision_result('Box', 'Cone', res)
            #
            # # req = fcl.DistanceRequest(enable_nearest_points=True)
            # # res = fcl.DistanceResult(min_distance_=0.1)
            #
            # # dist = fcl.distance(fcl.CollisionObject(mesh, fcl.Transform()),
            # #                     fcl.CollisionObject(mesh_hand, fcl.Transform()),
            # #                     req, res)
            # # print_distance_result('Box', 'Cone', res)
            #
            # # get contact points distance
            # # find the closest point on the mesh to each random point: https://trimsh.org/examples/nearest.html
            # distances, triangle_id, closest_points  = igl.point_mesh_squared_distance(contact_pts, hand_vertices, hand_faces)
            # print('Distance from point to surface of hand:\n{}\nclosest_points\n{}'.format(distances, closest_points))
            # for m, link in enumerate(contact_link):
            #     if link not in offset_contacts:
            #         offset_contacts[link]=[]
            #     offset_contacts[link].append(distances[m])
            # distances, triangle_id, closest_points  = igl.point_mesh_squared_distance(contact_pts, obj_verts, obj_faces)
            # print('Distance from point to surface of object:\n{}\nclosest_points\n{}'.format(distances, closest_points))
            # # hand_mesh = trimesh.Trimesh(vertices=hand_vertices,
            # #            faces=hand_faces)
            #
            # # mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
            # # with open(save_name,"w+") as fp:
            # #     fp.write(mesh_txt)
            if args.viz:
                plot_hand_w_object(obj_verts, obj_faces, hand_vertices, hand_faces, pts=contact_pts, jts=hand_joints, nmls=nmls, save_path=viz_dir + f'/{j}.png', viz_m=args.viz_m, viz_c=args.viz_c, viz_j=args.viz_j, viz_n=args.viz_n, save=True)
            joints_all[f'{instance}_{arti_ind}_{j}']   = hand_joints
            contacts_all[f'{instance}_{arti_ind}_{j}'] = contact_pts

    np.save(f'{grasps_meta}/{args.item}_joints.npy', joints_all)
    np.save(f'{grasps_meta}/{args.item}_contacts.npy', contacts_all)

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
