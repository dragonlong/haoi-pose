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

import __init__
from common.data_utils import fast_load_obj, sample_mesh, points_from_mesh
from utils.external.libmesh import check_mesh_contains

def create_ray_samples(sample_path,
                       min_hits=2000,
                       volumic=False,
                       cube_boundary=False,
                       display=False,
                       proj_based=False,
                       verbose=False):
    t0 = time.time()
    try:
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as obj_f:
                mesh_dict = pickle.load(obj_f)
        else:
            with open(sample_path.replace('.pkl', '.obj'), 'r') as obj_f:
                mesh_dict = fast_load_obj(obj_f)[0]
        print('Loaded {}'.format(sample_path))

        mesh = trimesh.load(mesh_dict)
        tri = Delaunay(mesh_dict['vertices'])
        # Now we have

        if display:
            dmesh = Poly3DCollection(
                mesh_dict['vertices'][tri.simplices[:, :3]], alpha=0.5)
            dmesh.set_edgecolor('b')
            dmesh.set_facecolor('r')
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(121, projection='3d')
            ax.add_collection3d(dmesh)

        if volumic:
            points = trimesh.sample.volume_mesh(mesh, count=min_hits)
            save_path = '/' + os.path.join(*sample_path.split('/')[:-1],
                                           'volume_points.pkl')
        elif proj_based:
            cloud  = get_surface_point_cloud(mesh, surface_point_method='scan', scan_count=20, scan_resolution=400)
            inds   = np.random.choice(cloud.points.shape[0], min_hits)
            points = cloud.points[inds]
            print('surface points have shape ', points.shape)
            save_path = '/' + os.path.join(*sample_path.split('/')[:-1],
                                           'surface_points.pkl')
        else:
            points = sample_mesh(mesh, min_hits=min_hits)
            if display:
                ax = fig.add_subplot(122, projection='3d')
                ax.scatter(points[:, 0], points[:, 1], points[:, 2])
                plt.show()
            save_path = '/' + os.path.join(*sample_path.split('/')[:-1], 'surface_points.pkl')

        with open(save_path, 'wb') as p_f:
            pickle.dump(points.astype(np.float16), p_f)

    except Exception:
        traceback.print_exc()
        if not volumic:
            obj_faces = np.array(mesh.faces)
            obj_verts3d = np.array(mesh.vertices)
            points = points_from_mesh(
                obj_faces, obj_verts3d, show_cloud=False, vertex_nb=min_hits)
            save_path = '/' + os.path.join(*sample_path.split('/')[:-1],
                                           'surface_points.pkl')
            print('Post_processing', save_path)
            with open(save_path, 'wb') as p_f:
                pickle.dump(points, p_f)
            print(class_id, sample)

    if cube_boundary:
        with open(sample_path.replace('.pkl', '_manifold.obj'), 'r') as obj_f:
            mesh_dict = fast_load_obj(obj_f)[0]
        print('Loaded {}'.format(sample_path.replace('.pkl', '_manifold.obj')))
        mesh = trimesh.load(mesh_dict)
        print('is_watertight: ', mesh.is_watertight)
        b_min = np.min(np.array(mesh.vertices), axis=0).reshape(1, 3) - 0.1
        b_max = np.max(np.array(mesh.vertices), axis=0).reshape(1, 3) + 0.1
        points =  np.random.rand(100000, 3) * max(1.1, np.max(b_max - b_min)) - 0.5 + (b_max + b_min)/2
        t0_occ = time.time()
        occupancies = check_mesh_contains(mesh, points)

        print(f'---occupancy for {points.shape[0]} pts takes {time.time()-t0_occ} sec')
        # S, I, C = igl.signed_distance(points, np.array(mesh.vertices), np.array(mesh.faces), return_normals=False)
        # points =  points[np.where(S>5e-5)[0]] # remove points on the surface
        # contain_helper = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        # occupancies = contain_helper.contains_points(points)
        save_path = '/' + os.path.join(*sample_path.split('/')[:-1], 'outside_points.pkl')
        with open(save_path, 'wb') as p_f:
            pts = points[~occupancies]
            # inds = np.random.choice(pts.shape[0], max(100000-points[occupancies].shape[0], 50000))
            # pickle.dump(pts[inds].astype(np.float16), p_f)
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
        if num< 1000:
            print(f'---too many inside points {num}, check!!!')

    if verbose:
        t1 = time.time()
        print(f'{min_hits} pts takes {t1-t0} sec')

def breakpoint():
    import pdb;pdb.set_trace()

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

if __name__ == "__main__":
    # selected_csv = '/sequoia/data2/dataset/shapenet/selected_atlas.csv'
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

    sample_paths = []
    for class_id, class_path in tqdm(shapenet_info.items(), desc='class'):
        samples = sorted(os.listdir(class_path))
        for sample in tqdm(samples, desc='sample'):
            sample_path = os.path.join(class_path, sample,
                                       'models/model_normalized.pkl')

            if class_id == '02958343' and (
                    sample == '207e69af994efa9330714334794526d4'):
                continue
            else:
                sample_paths.append(sample_path)

    print('Handling {} to {} from {} samples'.format(
        args.start_idx, args.start_idx + args.group_by, len(sample_paths)))

    for sample in tqdm(
            sample_paths[args.start_idx:args.start_idx + args.group_by]):
        t0 = time.time()

        # create water-tight mesh
        os.system('./manifold --input {} --output {}'.format(sample.replace('.pkl', '.obj'), sample.replace('.pkl', '_manifold.obj')))
        print(f'---create watertight mesh takes : {time.time()-t0} sec \n', sample.replace('.pkl', '_manifold.obj'))

        # save surface points
        create_ray_samples(sample, min_hits=100000, proj_based=True, cube_boundary=True, verbose=True)

        t1 = time.time()
        print(f'---totally take {t1-t0} seconds')
        print('')
