import argparse
import csv
import os
import time
import pickle
import traceback

import numpy as np

import sys
import random
import hydra
import h5py
import torch
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf
from os import makedirs, remove
from os.path import exists, join

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from tqdm import tqdm

import igl
import trimesh
import mesh_to_sdf
from mesh_to_sdf import get_surface_point_cloud
from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
from manopth.manolayer import ManoLayer

import pyrender
from scipy.spatial import Delaunay
from meshplot import plot, subplot, interact

import __init__
from global_info import global_info
from dataset import obman
from common.data_utils import fast_load_obj, sample_mesh, points_from_mesh
from common.vis_utils import visualize_mesh, plot_hand_w_object, visualize_2d, visualize_3d, plot2d_img
from common.queries import BaseQueries, TransQueries
from common import bp
from utils.external.libmesh import check_mesh_contains

infos       = global_info()
my_dir      = infos.base_path
second_path = infos.second_path
render_path = infos.render_path
platform_name = infos.platform_name
mano_path     = infos.mano_path
group_path = infos.group_path
epsilon = 10e-8
project_path= infos.project_path

def breakpoint():
    import pdb;pdb.set_trace()

# with open(f'{second_path}/data/hands/eyeglasses/0001/0/0.obj') as obj_f:
#     # file_path = '/home/dragon/Documents/external/ShapeNetCore.v2/02876657/1a7ba1f4c892e2da30711cdbdbc73924/models'
#     # with open(f'{file_path}/model_normalized.obj') as obj_f:
#     example_hand_mesh = trimesh.load(fast_load_obj(obj_f)[0])
#     hand_faces        = example_hand_mesh.faces
# points sampling
def create_ray_samples(hand_mesh,
                       obj_mesh,
                       sample_path,
                       obj_path,
                       min_hits=2000,
                       volumic_pts=False,
                       cube_pts_occupancy=False,
                       display=False,
                       surface_pts=False,
                       near_surface_sdf=False,
                       verbose=False):

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> check the pose stuff <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
    mesh = hand_mesh
    if surface_pts:
        cloud  = get_surface_point_cloud(mesh, surface_point_method='scan', scan_count=20, scan_resolution=400)
        inds   = np.random.choice(cloud.points.shape[0], min_hits)
        points = cloud.points[inds]
        hand_surface_points = np.copy(points)
        print('surface points have shape ', points.shape)
        save_path =  sample_path.replace('rgb', 'occupancy').replace('.jpg', '_hand_surface.pkl')
        # if display:
        #     visualize_mesh(mesh, pts=points, backend='pyrender', mode='trimesh', viz_mesh=False, title_name='surface_points')
        with open(save_path, 'wb') as p_f:
            pickle.dump(points.astype(np.float16), p_f)
    if cube_pts_occupancy:
        # with open(sample_path.replace('.pkl', '_manifold.obj'), 'r') as obj_f:
        #     mesh_dict_wt = fast_load_obj(obj_f)[0]
        # print('Loaded {}'.format(sample_path.replace('.pkl', '_manifold.obj')))
        # mesh = trimesh.load(mesh_dict_wt)
        print('is_watertight: ', mesh.is_watertight)
        # part 1: cube for hand, with object cube excluded
        b_min = np.min(np.array(obj_mesh.vertices), axis=0).reshape(1, 3) - 0.1
        b_max = np.max(np.array(obj_mesh.vertices), axis=0).reshape(1, 3) + 0.1
        corner_pt = (b_max + b_min)/2 - 0.5
        length_bb = max(1.1, np.max(b_max - b_min))

        b_hand_min = np.min(np.array(mesh.vertices), axis=0).reshape(1, 3) - 0.1
        b_hand_min = np.min(np.concatenate([b_min, b_hand_min], axis=0), axis=0)
        b_hand_max = np.max(np.array(mesh.vertices), axis=0).reshape(1, 3) + 0.1
        b_hand_max = np.max(np.concatenate([b_max, b_hand_max], axis=0), axis=0)
        points =  np.random.rand(200000, 3) * max(1.1, np.max(b_hand_max - b_hand_min)) - 0.5 + (b_hand_max + b_hand_min)/2
        # exclude points within object cube
        b_min = corner_pt
        b_max = corner_pt + np.array([[length_bb, length_bb, length_bb]])
        filter_idx = np.where((b_min[0, 0] > points[:, 0]) | (points[:, 0] > b_max[0, 0]) |
                              (b_min[0, 1] > points[:, 1]) | (points[:, 1] > b_max[0, 1]) |
                              (b_min[0, 2] > points[:, 2]) | (points[:, 2] > b_max[0, 2]))[0]
        points = points[filter_idx]
        if points.shape[0] > 100000:
            inds   = np.random.choice(points.shape[0], 100000)
            points = points[inds]

        # part 2: fetch object cube outside points
        obj_outside_path = '/' + os.path.join(*obj_path.split('/')[:-1], 'outside_points.pkl')
        obj_outside_points = np.load(obj_outside_path, allow_pickle=True)
        points = np.concatenate([obj_outside_points, points])

        t0_occ = time.time()
        occupancies = check_mesh_contains(mesh, points)
        print(f'---occupancy for {points.shape[0]} pts takes {time.time()-t0_occ} sec')

        save_path =  sample_path.replace('rgb', 'occupancy').replace('.jpg', '_hand_outside.pkl')
        with open(save_path, 'wb') as p_f:
            pts = points[~occupancies]
            if pts.shape[0] > 100000:
                inds= np.random.choice(pts.shape[0], 100000)
                pts = pts[inds]
            pickle.dump(pts.astype(np.float16), p_f)
        print('Saving to ', save_path, pts.shape[0], ' pts')

        # save_path =  sample_path.replace('rgb', 'occupancy').replace('.jpg', '_hand_full.pkl')
        # np.savez(save_path, outside_points=pts.astype(np.float16), inside_points=points[occupancies].astype(np.float16))
        # print('Saving to ', save_path)

        save_path =  sample_path.replace('rgb', 'occupancy').replace('.jpg', '_hand_inside.pkl')
        with open(save_path, 'wb') as p_f:
            pickle.dump(points[occupancies].astype(np.float16), p_f)
        print('Saving to ', save_path, points[occupancies].shape[0], ' pts')
        num = points[occupancies].shape[0]
        hand_inside_points = np.copy(points[occupancies])
        if display:
            # inner_center = np.mean(points[occupancies], axis=0)
            inner_center = points[occupancies][1, :]
            print(np.min( np.abs(points[~occupancies] - inner_center.reshape(1, 3)), axis=0))
            # visualize_mesh(mesh, pts=points, labels=occupancies, mode='trimesh', backend='pyrender', viz_mesh=True, title_name='full_points')
            visualize_mesh(mesh, pts=points[~occupancies], backend='pyrender', mode='trimesh', viz_mesh=False, title_name='outside_points')
            visualize_mesh(mesh, pts=points[occupancies], backend='pyrender', mode='trimesh', viz_mesh=False, title_name='inside_points')

    # check object points intersect with hand:
    check_contacts = True
    if check_contacts:
        obj_inside_path = '/' + os.path.join(*obj_path.split('/')[:-1], 'inside_points.pkl')
        obj_inside_points = np.load(obj_inside_path, allow_pickle=True)
        obj_surface_path = '/' + os.path.join(*obj_path.split('/')[:-1], 'surface_points.pkl')
        obj_surface_points= np.load(obj_surface_path, allow_pickle=True)
        S, I, C = igl.signed_distance(obj_surface_points.astype(np.float32), np.array(mesh.vertices), np.array(mesh.faces), return_normals=False)
        contacts_index = np.where(S<0.0005)[0]
        if len(contacts_index) > 0:
            contact_pts = obj_surface_points[contacts_index]
        if len(contacts_index) < 100 and len(contacts_index) > 0:
        # # hand pts
        # # S, I, C = igl.signed_distance(hand_surface_points.astype(np.float32), np.array(obj_mesh.vertices), np.array(obj_mesh.faces), return_normals=False)
        # with open(obj_path.replace('.pkl', '_manifold.obj'), 'r') as obj_f:
        #     mesh_dict_wt = fast_load_obj(obj_f)[0]
        # print('Loaded {}'.format(obj_path.replace('.pkl', '_manifold.obj')))
        # obj_mesh = trimesh.load(mesh_dict_wt)
        # S, I, C = igl.signed_distance(hand_surface_points.astype(np.float32), np.array(obj_mesh.vertices), np.array(obj_mesh.faces), return_normals=False)
        # hand_contacts_index = np.where(S<0.0005)[0]
        # if len(hand_contacts_index) > 0:
        #     hand_contact_pts = hand_surface_points[hand_contacts_index]
            dis = np.linalg.norm(hand_surface_points[:, np.newaxis, :] - contact_pts[np.newaxis,:, :], axis=-1)
            hand_contacts_index = np.where(np.min(dis, axis=1) < 0.0002)[0]
        else:
            hand_contacts_index = []
        if len(hand_contacts_index) > 0:
            hand_contact_pts = hand_surface_points[hand_contacts_index]
        save_path = sample_path.replace('rgb', 'occupancy').replace('.jpg', '_contacts.npz')
        if len(contacts_index) > 0 and len(hand_contacts_index) > 0:
            np.savez(save_path, hand=hand_contacts_index.astype(np.int16), hand_pts=hand_contact_pts.astype(np.float16), object=contacts_index.astype(np.int16), object_pts=contact_pts.astype(np.float16))
        else:
            np.savez(save_path, null=np.array([0]))
        print('Saving to ', save_path)
        if display:
            print(f'we have {len(contacts_index)} obj contact pts')
            print(f'we have {len(hand_contacts_index)} hand contact pts')
            visualize_mesh([mesh, obj_mesh], pts=contact_pts, mode='trimesh', backend='pyrender', viz_mesh=True, title_name='full_points')
            visualize_mesh([mesh, obj_mesh], pts=hand_contact_pts, mode='trimesh', backend='pyrender', viz_mesh=True, title_name='full_points')

    if near_surface_sdf:
        t0 = time.time()
        cloud  = get_surface_point_cloud(mesh, surface_point_method='scan', scan_count=20, scan_resolution=400, sample_point_count=100000)
        t1 = time.time()
        print(f'surface pts: {t1-t0} sec for {cloud.points.shape[0]} pts')
        # visualize_mesh(mesh_dict, pts=cloud.points, backend='pyrender', viz_mesh=False, title_name='full_points')
        t1 = time.time()
        # near surface pts
        points = np.copy(cloud.points) + (np.random.rand(cloud.points.shape[0], 3) - 0.5) * 0.1
        inds   = np.random.choice(points.shape[0], 100000)
        points = points[inds]
        # far way points
        with open(sample_path.replace('.pkl', '_manifold.obj'), 'r') as obj_f:
            mesh_dict_wt = fast_load_obj(obj_f)[0]
        print('Loaded {}'.format(sample_path.replace('.pkl', '_manifold.obj')))
        mesh = trimesh.load(mesh_dict_wt)
        print('is_watertight: ', mesh.is_watertight)
        b_min = np.min(np.array(mesh.vertices), axis=0).reshape(1, 3) - 0.1
        b_max = np.max(np.array(mesh.vertices), axis=0).reshape(1, 3) + 0.1
        points1 =  np.random.rand(300000, 3) * max(1.1, np.max(b_max - b_min)) - 0.5 + (b_max + b_min)/2
        points = np.concatenate([points, points1], axis=0)

        S, I, C = igl.signed_distance(points, np.array(mesh.vertices), np.array(mesh.faces), return_normals=False)
        t2 = time.time()
        print(f'igl SDF:  {t2-t1} sec for {points.shape[0]} pts')
        occupancies = check_mesh_contains(mesh, points)
        t3 = time.time()
        print(f'Occupancy checking: {t3-t2} sec for {points.shape[0]} pts')
        S[~occupancies] = - S[~occupancies]
        # visualize_mesh(mesh_dict, pts=points[occupancies], backend='pyrender', viz_mesh=True, title_name='inside_points')
        # visualize_mesh(mesh_dict, pts=points[~occupancies], backend='pyrender', viz_mesh=False, title_name='outside_points')

        save_path = '/' + os.path.join(*sample_path.split('/')[:-1], 'sdf_points.npz')
        np.savez(save_path, near_points=points.astype(np.float16), sdf_value=S.astype(np.float16))
        print('Saving to ', save_path)

    if verbose:
        t1 = time.time()
        print(f'{min_hits} pts takes {t1-t0} sec')

# hand_joints2d = pose_dataset.get_joints2d(img_idx)
# hand_verts2d  = pose_dataset.get_verts2d(img_idx)
# hand_joints2d = pose_dataset.get_joints2d(img_idx)
# obj_verts2d = pose_dataset.get_obj_verts2d(img_idx)
# add extra faces
#
# visualize_2d(
#     img,
#     hand_joints=hand_joints2d,
#     hand_verts=hand_verts2d,
#     obj_verts=obj_verts2d)
# visualize_3d(
#     img,
#     hand_verts=hand_verts3d,
#     hand_faces=hand_faces,
#     obj_verts=obj_verts3d,
#     obj_faces=obj_faces)

def get_dataset(cfg,
                dset_name,
                split,
                train_it=True,
                black_padding=False,
                max_queries=None,
                use_cache=True):
    # begin here
    task=cfg.task
    meta={
        "mode": cfg.DATASET.mode,
        "override_scale": cfg.DATASET.override_scale,
        "fhbhands_split_type": cfg.DATASET.fhbhands_split_type,
        "fhbhands_split_choice": cfg.DATASET.fhbhands_split_choice,
        "fhbhands_topology": cfg.DATASET.fhbhands_topology,
    }
    max_queries=max_queries
    point_nb=cfg.DATASET.atlas_points_nb
    center_idx=cfg.DATASET.center_idx
    limit_size=cfg.limit_size
    sides=cfg.DATASET.sides

    pose_dataset = obman.ObMan(
        split=split,
        use_cache=use_cache,
        mode=meta["mode"],
        mini_factor=cfg.DATASET.mini_factor,
        override_scale=meta["override_scale"],
        segment=False,
        use_external_points=True,
        apply_obj_transform=False,
        shapenet_root=f"{group_path}/external/ShapeNetCore.v2",
        obman_root=f"{group_path}/external/obman/obman")

    return pose_dataset

def get_queries(cfg):
    max_queries = [
        BaseQueries.occupancy,
        BaseQueries.depth,
        BaseQueries.pcloud,
        BaseQueries.nocs,
        TransQueries.nocs,
    ]
    # max_queries = [
    #     TransQueries.affinetrans,
    #     TransQueries.images,
    #     TransQueries.verts3d,
    #     TransQueries.center3d,
    #     TransQueries.joints3d,
    #     TransQueries.objpoints3d,
    #     TransQueries.pcloud,
    #     TransQueries.camintrs,
    #     BaseQueries.sides,
    #     BaseQueries.camintrs,
    #     BaseQueries.meta,
    #     BaseQueries.segms,
    #     BaseQueries.depth,
    #     TransQueries.sdf,
    #     TransQueries.sdf_points,
    # ]
    if cfg.DATASET.mano_lambda_joints2d:
        max_queries.append(TransQueries.joints2d)
    return max_queries

@hydra.main(config_path="../config/occupancy.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    random.seed(30)

    limit_size = cfg.limit_size
    dset_name  = 'obman'
    max_queries= get_queries(cfg)

    # train_dataset = get_dataset(
    #     cfg,
    #     dset_name,
    #     max_queries=max_queries,
    #     split='train',
    #     train_it=True)

    valid_dataset = get_dataset(cfg,
        dset_name,
        max_queries=max_queries,
        split='val',
        train_it=False)
    pose_dataset = valid_dataset
    batch_num = int(len(pose_dataset) / cfg.num) + 1
    start_ind = batch_num * cfg.idx
    end_ind   = min(batch_num * (cfg.idx + 1) + 1, len(pose_dataset))
    for img_idx in range(start_ind, end_ind):
        img = pose_dataset.get_image(img_idx)
        side = pose_dataset.get_sides(img_idx)
        hand_verts3d = pose_dataset.get_verts3d(img_idx)
        hand_faces   = pose_dataset.get_faces3d(img_idx)
        extra_faces  = np.array([[78, 121, 79],
                                [121, 214, 79],
                                [79, 214, 108],
                                [108, 214, 120],
                                [120, 214, 215],
                                [215, 119, 120],
                                [119, 215, 279],
                                [279, 117, 119],
                                [117, 279, 239],
                                [239, 118, 117],
                                [118, 239, 234],
                                [234, 122, 118],
                                [122, 234, 92],
                                [92, 38, 122],
                                ])
        hand_faces = np.concatenate([hand_faces, extra_faces], axis=0)
        obj_verts3d, obj_faces, canon_pts = pose_dataset.get_obj_verts_faces(img_idx)
        hand_mesh = trimesh.Trimesh(vertices=hand_verts3d, faces=hand_faces, process=False)
        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        obj_mesh  = trimesh.Trimesh(vertices=obj_verts3d, faces=obj_faces, process=False)

        save_hand_mesh = False
        if save_hand_mesh:
            mesh_txt = trimesh.exchange.obj.export_obj(hand_mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
            save_dir = './' # TODO
            save_name= save_dir + f'hand_{side}.obj'
            with open(save_name,"w+") as fp:
                fp.write(mesh_txt)

        use_depth_nocs = True
        if use_depth_nocs:
            depth = pose_dataset.get_depth(img_idx)
            camintr = pose_dataset.get_camintr(img_idx)
            cloud = pose_dataset.get_pcloud(depth, camintr) # * 1000 - center3d
            obj_hand_segm = (np.asarray(pose_dataset.get_segm(img_idx, debug=False)) / 255).astype(np.int)
            full_segm = obj_hand_segm[:, :, 0] | obj_hand_segm[:, :, 1]
            segm = full_segm
            pts  = cloud[np.where(segm>0)[0], np.where(segm>0)[1], :]
            nocs = pose_dataset.get_nocs(img_idx, pts, boundary_pts)

        sample_path = pose_dataset.image_names[img_idx]
        obj_path    = pose_dataset.obj_paths[img_idx]
        occupancy_path =  os.path.join(pose_dataset.obman_root, 'occupancy')
        if not exists(occupancy_path):
            makedirs(occupancy_path)

        t0 = time.time()
        # create_ray_samples(hand_mesh, obj_mesh, sample_path, obj_path, cube_pts_occupancy=True, display=True, surface_pts=True, near_surface_sdf=False, min_hits=100000)
        print(f'--- {img_idx}th data: {time.time() - t0} seconds')
        display = True
        if display:
            # plot depth image
            # plot2d_img([depth*250.0], title_name='depth', dpi=300, show_fig=True)
            # plot_hand_w_object(hand_verts3d, hand_faces, obj_verts3d, obj_faces)
            scene = pyrender.Scene()
            # mesh_vis = pyrender.Mesh.from_trimesh(hand_mesh)
            # scene.add(mesh_vis)
            # # object
            # mesh_vis = pyrender.Mesh.from_trimesh(obj_mesh)
            # scene.add(mesh_vis)
            # # pts = nocs
            pts = np.concatenate([canon_pts, hand_verts3d], axis=0)
            sm = trimesh.creation.uv_sphere(radius=0.002)
            sm.visual.vertex_colors = [1.0, 1.0, 0.0]
            tfs = np.tile(np.eye(4), (len(pts), 1, 1))
            tfs[:,:3,3] = pts
            m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(m)
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, show_world_axis=False, window_title='hand mesh')

if __name__ == '__main__':
    main()
