import argparse
import csv
import os
import time
import pickle
import traceback

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
import igl
from tqdm import tqdm
from scipy.spatial import Delaunay
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import mesh_to_sdf
from mesh_to_sdf import get_surface_point_cloud
import igl
from meshplot import plot, subplot, interact

import __init__
from common.data_utils import fast_load_obj, sample_mesh, points_from_mesh
from common.vis_utils import visualize_mesh
from common import bp
from utils.external.libmesh import check_mesh_contains
from global_info import global_info

infos     = global_info()
my_dir    = infos.base_path
second_path = infos.second_path
render_path = infos.render_path
platform_name = infos.platform_name

# points sampling
def create_ray_samples(sample_path,
                       min_hits=2000,
                       volumic_pts=False,
                       cube_pts_occupancy=False,
                       display=False,
                       surface_pts=False,
                       near_surface_sdf=False,
                       verbose=False):
    t0 = time.time()
    # try:
    if os.path.exists(sample_path):
        with open(sample_path, 'rb') as obj_f:
            mesh_dict = pickle.load(obj_f)
    else:
        with open(sample_path.replace('.pkl', '.obj'), 'r') as obj_f:
            mesh_dict = fast_load_obj(obj_f)[0]
    print('Loaded {}'.format(sample_path))
    mesh = trimesh.load(mesh_dict)
    # if volumic_pts: # get points inside
    #     points = trimesh.sample.volume_mesh(mesh, count=min_hits)
    #     save_path = '/' + os.path.join(*sample_path.split('/')[:-1], 'volume_points.pkl')
    #     visualize_mesh(mesh_dict, pts=points, backend='pyrender', viz_mesh=False)
    #     with open(save_path, 'wb') as p_f:
    #         pickle.dump(points.astype(np.float16), p_f)

    if surface_pts:
        cloud  = get_surface_point_cloud(mesh, surface_point_method='scan', scan_count=20, scan_resolution=400)
        inds   = np.random.choice(cloud.points.shape[0], min_hits)
        points = cloud.points[inds]
        print('surface points have shape ', points.shape)
        save_path = '/' + os.path.join(*sample_path.split('/')[:-1],
                                       'surface_points.pkl')
        # points_from_mesh()
        with open(save_path, 'wb') as p_f:
            pickle.dump(points.astype(np.float16), p_f)
        if display:
            visualize_mesh(mesh_dict, pts=points, backend='pyrender', viz_mesh=False, title_name='surface_points')

    if cube_pts_occupancy: #
        with open(sample_path.replace('.pkl', '_manifold.obj'), 'r') as obj_f:
            mesh_dict_wt = fast_load_obj(obj_f)[0]
        print('Loaded {}'.format(sample_path.replace('.pkl', '_manifold.obj')))
        mesh = trimesh.load(mesh_dict_wt)
        print('is_watertight: ', mesh.is_watertight)
        b_min = np.min(np.array(mesh.vertices), axis=0).reshape(1, 3) - 0.1
        b_max = np.max(np.array(mesh.vertices), axis=0).reshape(1, 3) + 0.1
        points =  np.random.rand(100000, 3) * max(1.1, np.max(b_max - b_min)) - 0.5 + (b_max + b_min)/2
        t0_occ = time.time()
        occupancies = check_mesh_contains(mesh, points)

        print(f'---occupancy for {points.shape[0]} pts takes {time.time()-t0_occ} sec')

        save_path = '/' + os.path.join(*sample_path.split('/')[:-1], 'outside_points.pkl')
        with open(save_path, 'wb') as p_f:
            pts = points[~occupancies]
            pickle.dump(pts.astype(np.float16), p_f)
        print('Saving to ', save_path, pts.shape[0], ' pts')

        save_path = '/' + os.path.join(*sample_path.split('/')[:-1], 'full_points.npz')
        np.savez(save_path, outside_points=pts.astype(np.float16), inside_points=points[occupancies].astype(np.float16))
        print('Saving to ', save_path)

        save_path = '/' + os.path.join(*sample_path.split('/')[:-1], 'inside_points.pkl')
        with open(save_path, 'wb') as p_f:
            pickle.dump(points[occupancies].astype(np.float16), p_f)
        print('Saving to ', save_path, points[occupancies].shape[0], ' pts')
        num = points[occupancies].shape[0]
        if display:
            visualize_mesh(mesh_dict, pts=points, labels=occupancies, backend='pyrender', viz_mesh=False, title_name='full_points')
            visualize_mesh(mesh_dict, pts=points[~occupancies], backend='pyrender', viz_mesh=False, title_name='outside_points')
            visualize_mesh(mesh_dict, pts=points[occupancies], backend='pyrender', viz_mesh=False, title_name='inside_points')

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

    if save_hand_sdf:
        # fetch hadn mesh plus object mesh scaling factors;

        # transform into canonical

        # re-sample surface points

        # re-check occupancy values for those cube points for both objects & hands

    if verbose:
        t1 = time.time()
        print(f'{min_hits} pts takes {t1-t0} sec')

def breakpoint():
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_by', default=4000, type=int)
    parser.add_argument('--start_idx', default=0, type=int)
    args = parser.parse_args()
    selected_csv = '../scripts/shapenet_select.csv'
    shapenet_info = {}
    with open(selected_csv, 'r') as csv_f:
        reader = csv.DictReader(csv_f)
        for row_idx, row in enumerate(reader):
            shapenet_info[row['class']] = row['path']

    if platform_name == 'dragon':
        category_path = second_path + f'/external/ShapeNetCore.v2/02876657'
        sample_paths = [os.path.join(category_path, sample, 'models/model_normalized.pkl') for sample in os.listdir(category_path)]
    else:
        sample_paths = []
        for class_id, class_path in tqdm(shapenet_info.items(), desc='class'):
            samples = sorted(os.listdir(class_path))
            for sample in tqdm(samples, desc='sample'):
                sample_path = os.path.join(class_path, sample, 'models/model_normalized.pkl')

                if class_id == '02958343' and (
                        sample == '207e69af994efa9330714334794526d4'):
                    continue
                else:
                    sample_paths.append(sample_path)

    print('Handling {} to {} from {} samples'.format(args.start_idx, args.start_idx + args.group_by, len(sample_paths)))
    for sample in tqdm(sample_paths[args.start_idx:args.start_idx + args.group_by]):
        t0 = time.time()
        # create water-tight mesh
        if not os.path.exists(sample.replace('.pkl', '_manifold.obj')):
            os.system('./manifold --input {} --output {}'.format(sample.replace('.pkl', '.obj'), sample.replace('.pkl', '_manifold.obj')))
            print(f'---create watertight mesh takes : {time.time()-t0} sec \n', sample.replace('.pkl', '_manifold.obj'))

        # save surface points
        create_ray_samples(sample, min_hits=100000, volumic_pts=False, surface_pts=False, cube_pts_occupancy=False, near_surface_sdf=True, verbose=True, display=False)

        t1 = time.time()
        print(f'---totally take {t1-t0} seconds')
        print('')
