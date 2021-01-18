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
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import mesh_to_sdf
from mesh_to_sdf import get_surface_point_cloud
import igl
from meshplot import plot, subplot, interact

import __init__
from common.data_utils import fast_load_obj, sample_mesh, points_from_mesh
from common.vis_utils import visualize_mesh
from common.d3_utils import rotate_about_axis
from common import bp
from utils.external.libmesh import check_mesh_contains
from global_info import global_info

infos     = global_info()
my_dir    = infos.base_path
second_path = infos.second_path
render_path = infos.render_path
platform_name = infos.platform_name
project_path = infos.project_path
symmetry_dict= infos.symmetry_dict

def bp():
    import pdb;pdb.set_trace()

# points sampling
def check_symmetry_sample(sample_path,
                       display=False,
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

    vertices =mesh.vertices

    center = np.sum(vertices, axis=0)
    print('center1: ', center)
    center_condition1 = abs(center[0]) < 1e-3 and abs(center[2]) < 1e-3
    boundary_max= np.max(vertices, axis=0)
    boundary_min= np.min(vertices, axis=0)
    lengths = boundary_max - boundary_min
    center_pt = (boundary_max + boundary_min)/2
    center = center - center_pt
    print('center2: ', center)
    print('lengths: ', lengths)
    center_condition2 = abs(center[0]) < 1e-2 and abs(center[2]) < 1e-2

    step1 = center_condition2 and abs(lengths[0] - lengths[2]) < 1e-3
    if center_condition2 and abs(lengths[0] - lengths[2]) < 1e-3:
        print('---step 1 symmetric!!!')
    else:
        print('---step 1 Not symmetric!!!')

    # further rotate 45 degrees
    vertices = vertices - center_pt
    rot_mat = rotate_about_axis(np.pi/4, axis='y')
    vertices = np.dot(vertices, rot_mat)
    boundary_max= np.max(vertices, axis=0)
    boundary_min= np.min(vertices, axis=0)
    lengths2 = boundary_max - boundary_min
    center_pt = (boundary_max + boundary_min)/2
    center = center - center_pt
    print('center2: ', center)
    print('lengths: ', lengths2)
    center_condition2 = abs(center[0]) < 1e-2 and abs(center[2]) < 1e-2
    step2 = center_condition2 and abs(lengths2[0] - lengths2[2]) < 1e-3 and abs(lengths[0] - lengths2[0]) < 1e-3
    if center_condition2 and abs(lengths2[0] - lengths2[2]) < 1e-3 and abs(lengths[0] - lengths2[0]) < 1e-3:
        print('------step 2 symmetric!!!')
    else:
        print('---step 2 Not symmetric!!!')

    if display:
        visualize_mesh(mesh, backend='pyrender', mode='trimesh', viz_mesh=True, title_name='mesh_points')

    if step1 and step2:
        return True
    else:
        return False

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
        category_path = second_path + f'/../external/ShapeNetCore.v2/02876657'
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
    symmetry_dict = {}
    for sample in tqdm(sample_paths[args.start_idx:args.start_idx + args.group_by]):
        t0 = time.time()
        # save surface points
        is_symmetric = check_symmetry_sample(sample, verbose=True, display=False)
        class_id, instance_id = sample.split('/')[-4:-2]
        symmetry_dict[f'{class_id}_{instance_id}'] = is_symmetric
        t1 = time.time()
        print(f'---totally take {t1-t0} seconds')
        print('')
    # save locally
    np.save(f'{project_path}/haoi-pose/dataset/data/symmetry.npy', symmetry_dict)
