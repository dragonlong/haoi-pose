import numpy as np
import os
from os import makedirs, remove
from os.path import exists, join
import numpy as np
from numpy.linalg import matrix_rank, inv
from plyfile import PlyData, PlyElement

COLOR_MAP_RGB =  [[0, 0, 0],
                [0, 0, 255],
                [245, 150, 100],
                [245, 230, 100],
                [250, 80, 100],
                [150, 60, 30],
                [255, 0, 0],
                [180, 30, 80],
                [255, 0, 0],
                [30, 30, 255],
                [200, 40, 255],
                [90, 30, 150],
                [255, 0, 255],
                [255, 150, 255],
                [75, 0, 75],
                [75, 0, 175],
                [0, 200, 255],
                [50, 120, 255],
                [0, 150, 255],
                [170, 255, 150],
                [0, 175, 0],
                [0, 60, 135],
                [80, 240, 150],
                [150, 240, 255],
                [0, 0, 255],
                [255, 255, 50],
                [245, 150, 100],
                [255, 0, 0],
                [200, 40, 255],
                [30, 30, 255],
                [90, 30, 150],
                [250, 80, 100],
                [180, 30, 80],
                [255, 0, 0]]
#
def breakpoint():
    import pdb; pdb.set_trace()

def bp():
    import pdb; pdb.set_trace()

def dist_tester():
    print('please use ')

def print_group(values, names=None):
    if names is None:
        for j in range(len(values)):
            print(values[j], '\n')
    else:
        for j in range(len(names)):
            print(names[j], '\n', values[j], '\n')

def colorize_pointcloud(xyz, label, ignore_label=255):
  assert label[label != ignore_label].max() < len(COLOR_MAP_RGB), 'Not enough colors.'
  label_rgb = np.array([COLOR_MAP_RGB[i] if i != ignore_label else IGNORE_COLOR for i in label])
  return np.hstack((xyz, label_rgb))

def read_plyfile(filepath):
  """Read ply file and return it as numpy array. Returns None if emtpy."""
  with open(filepath, 'rb') as f:
    plydata = PlyData.read(f)
  if plydata.elements:
    return pd.DataFrame(plydata.elements[0].data).values


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
  """Save an RGB point cloud as a PLY file.

  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
  assert points_3d.ndim == 2
  if with_label:
    assert points_3d.shape[1] == 7
    python_types = (float, float, float, int, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1'), ('label', 'u1')]
  else:
    if points_3d.shape[1] == 3:
      gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
      points_3d = np.hstack((points_3d, gray_concat))
    assert points_3d.shape[1] == 6
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
  if binary is True:
    # Format into NumPy structured array
    vertices = []
    for row_idx in range(points_3d.shape[0]):
      cur_point = points_3d[row_idx]
      vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    # Write
    PlyData([el]).write(filename)
  else:
    # PlyData([el], text=True).write(filename)
    with open(filename, 'w') as f:
      f.write('ply\n'
              'format ascii 1.0\n'
              'element vertex %d\n'
              'property float x\n'
              'property float y\n'
              'property float z\n'
              'property uchar red\n'
              'property uchar green\n'
              'property uchar blue\n'
              'property uchar alpha\n'
              'end_header\n' % points_3d.shape[0])
      for row_idx in range(points_3d.shape[0]):
        X, Y, Z, R, G, B = points_3d[row_idx]
        f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
  if verbose is True:
    print('Saved point cloud to: %s' % filename)

# def print_group(names, values):
#     for subname, subvalue in zip(names, values):
#         print(subname, ':\n', subvalue)
#         print('')
# # np.savetxt(f'{save_path}/contact_err_{domain}.txt', contacts_err, fmt='%1.5f', delimiter='\n')

def check_h5_keys(hf, verbose=False):
    print('')
    if verbose:
        for key in list(hf.keys()):
            print(key, hf[key][()].shape)

def save_for_viz(names, objects, save_name='default.npy', type='tensor', verbose=True):
    # names = ['coords', 'input', 'target']
    # objects = [coords, input, target]
    save_input_data = True
    print('---saving to ', save_name)
    if save_input_data:
        viz_dict = {}
        for name, item in zip(names, objects):
            if type == 'tensor':
                viz_dict[name] = item.cpu().numpy()
            else:
                viz_dict[name] = item
            if verbose:
                try:
                    print(name, ': ', item.shape)
                except:
                    print(name, ': ', item)
        # save_name = f'{self.config.save_viz_dir}/input_{iteration}.npy'
        save_path = os.path.dirname(save_name)
        if not exists(save_path):
            makedirs(save_path)
        if len(objects) == 2 and 'labels' in names and 'points' in names:
            xyzs = colorize_pointcloud(viz_dict['points'][:, :3], viz_dict['labels'].astype(np.int8))
            save_point_cloud(xyzs, save_name.replace('npy', 'ply'), with_label=False)
        else:
            np.save(save_name, arr=viz_dict)

def visualize_results(coords, input, target, upsampled_pred, config, iteration):
  # Get filter for valid predictions in the first batch.
  target_batch = coords[:, 3].numpy() == 0
  input_xyz = coords[:, :3].numpy()
  target_valid = target.numpy() != 255
  target_pred = np.logical_and(target_batch, target_valid)
  target_nonpred = np.logical_and(target_batch, ~target_valid)
  ptc_nonpred = np.hstack((input_xyz[target_nonpred], np.zeros((np.sum(target_nonpred), 3))))
  # Unwrap file index if tested with rotation.
  file_iter = iteration
  if config.test_rotation >= 1:
    file_iter = iteration // config.test_rotation
  # Create directory to save visualization results.
  os.makedirs(config.visualize_path, exist_ok=True)
  # Label visualization in RGB.
  xyzlabel = colorize_pointcloud(input_xyz[target_pred], upsampled_pred[target_pred])
  xyzlabel = np.vstack((xyzlabel, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'pred', '%04d.ply' % file_iter])
  save_point_cloud(xyzlabel, os.path.join(config.visualize_path, filename), verbose=False)
  # RGB input values visualization.
  xyzrgb = np.hstack((input_xyz[target_batch], input[:, :3].cpu().numpy()[target_batch]))
  filename = '_'.join([config.dataset, config.model, 'rgb', '%04d.ply' % file_iter])
  save_point_cloud(xyzrgb, os.path.join(config.visualize_path, filename), verbose=False)
  # Ground-truth visualization in RGB.
  xyzgt = colorize_pointcloud(input_xyz[target_pred], target.numpy()[target_pred])
  xyzgt = np.vstack((xyzgt, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'gt', '%04d.ply' % file_iter])
  save_point_cloud(xyzgt, os.path.join(config.visualize_path, filename), verbose=False)
