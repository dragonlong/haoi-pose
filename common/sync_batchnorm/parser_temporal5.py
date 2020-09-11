import os
import yaml
import numpy as np
import time
import math
import h5py
import torch
from torch.utils.data import Dataset
from random import randint, sample
from numpy.linalg import inv
import matplotlib
# matplotlib.use('Agg') # TkAgg
import matplotlib.pyplot as plt  #
from multiprocessing import Manager

import __init__ as booger
from config import cfg
from common.laserscan import LaserScan, SemLaserScan
from common.data_utils import voxelize_occupy, gen_2d_grid_gt_from_pts, gen_3d_voxel_gt_from_pts, save_h5_bev, voxelize_occupy_detail, parse_calibration, parse_poses
from common.vis_utils import plot2d_img
from modules.ioueval import iouEval
EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL= ['.label']
EXTENSIONS_IND= ['.npy']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def is_npy(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IND)

def plot_imgs(imgs, imgs_name, title_name='default', sub_name='default', save_path=None, save_fig=False, axis_off=False, grid_on=False, show_fig=True, dpi=150):
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    num = len(imgs)
    for m in range(num):
        ax1 = plt.subplot(1, num, m+1)
        plt.imshow(imgs[m].astype(np.uint8))
        if title_name is not None:
            plt.title(f'{title_name} ' + r'$_{{ {t2} }}$'.format(t2=imgs_name[m]))
        else:
            plt.title(imgs_name[m])
        if grid_on:
          plt.grid('on')
    if show_fig:
        plt.show()
    if save_fig:
        if save_path is None:
            if not os.path.exists('./visualizations/'):
                os.makedirs('./visualizations/')
            fig.savefig('./visualizations/{}_{}.png'.format(sub_name, title_name), pad_inches=0)
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig('{}/{}_{}.png'.format(save_path, sub_name, title_name), pad_inches=0)
    plt.close()

def plot_imgsup(imgs, imgs_name, title_name='default', sub_name='default', save_path=None, save_fig=False, axis_off=False, show_fig=True, dpi=150):
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    num = len(imgs)
    for m in range(num):
        ax1 = plt.subplot(num, 1, m+1)
        plt.imshow(imgs[m].astype(np.uint8))
        if title_name is not None:
            plt.title(title_name[0]+ ' ' + imgs_name[m])
        else:
            plt.title(imgs_name[m])
    if show_fig:
        plt.show()

    plt.close()

class SemanticKitti(Dataset):
  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               cfg,              # sensor to parse scans from
               max_points=150000, # max number of points present in dataset
               past_frame_skip=1, # use for similar data skipping
               future_frame_skip=1, # use
               seq_length=2,
               label_dim=3,   # dimension for the label
               gt=True,
               is_train=True):      # send ground truth?
    # save deats
    # print(root)
    self.root = os.path.join(root, "sequences")
    self.save_preprocess_path = os.path.join(root, 'preprocessed_seq5')
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.cfg = cfg
    self.is_train= is_train
    self.augment = cfg.DATASET.augment
    self.sensor_img_H = cfg.DATASET.img_height
    self.sensor_img_W = cfg.DATASET.img_width
    self.sensor_img_means = torch.tensor(cfg.DATASET.img_means,
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(cfg.DATASET.img_stds,
                                        dtype=torch.float)
    self.sensor_fov_up = cfg.DATASET.proj_fov_up
    self.sensor_fov_down = cfg.DATASET.proj_fov_down
    self.max_points = max_points
    self.use_rv=cfg.TRAIN.use_rv    # whether to use range view data
    self.use_bev=cfg.TRAIN.use_bev  # whether to use bev data
    self.gt = gt
    self.compute_on_the_fly = False    # only true during testing stage
    self.label_dim   = cfg.BEV.label_dim
    print('label_dim is {}d'.format(self.label_dim))
    self.voxel_size  =np.array(cfg.DATASET.voxel_size)# voxel size in meter
    self.area_extents=np.array(cfg.DATASET.area_extents)

    # so the number that matters is how many there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # voxel
    self.past_frame_skip = past_frame_skip
    self.future_frame_skip = future_frame_skip
    self.num_sweep = seq_length
    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files  = []
    self.label_files = []
    self.indice_files= []
    self.seq_num_dict = {}

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      print("parsing seq {}".format(seq))

      # get paths for each
      scan_path  = os.path.join(self.root, seq, "velodyne")
      label_path = os.path.join(self.root, seq, "labels")
      indice_path = os.path.join(root, 'preprocessed', seq)
      print('we have preprocessed path: ', indice_path)

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]
      indice_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(indice_path)) for f in fn if is_npy(f)]
      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # extend list
      self.scan_files.extend(scan_files)
      self.label_files.extend(label_files)
      self.indice_files.extend(indice_files)

    # sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()
    self.indice_files.sort()

    if self.future_frame_skip > 1:
      self.scan_files = self.scan_files[::self.future_frame_skip]
      self.label_files = self.label_files[::self.future_frame_skip]
      self.indice_files = self.indice_files[::self.future_frame_skip]
    print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                    self.sequences))
    manager = Manager()
    self.cache = manager.dict()
    self.motion_map = {
              0 : 0,     # "unlabeled"
              1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
              10: 1,     # "car"
              11: 1,     # "bicycle"
              13: 1,     # "bus" mapped to "other-vehicle" --------------------------mapped
              15: 1,     # "motorcycle"
              16: 1,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
              18: 1,     # "truck"
              20: 1,     # "other-vehicle"
              30: 1,     # "person"
              31: 1,     # "bicyclist"
              32: 1,     # "motorcyclist"
              40: 0,     # "road"
              44: 0,    # "parking"
              48: 0,    # "sidewalk"
              49: 0,    # "other-ground"
              50: 0,    # "building"
              51: 0,    # "fence"
              52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
              60: 0,     # "lane-marking" to "road" ---------------------------------mapped
              70: 0,    # "vegetation"
              71: 0,    # "trunk"
              72: 0,    # "terrain"
              80: 0,    # "pole"
              81: 0,    # "traffic-sign"
              99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
              252: 2,    # "moving-bicyclist"
              254: 2,    # "moving-person"
              255: 2,    # "moving-motorcyclist"
              256: 2,    # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
              257: 2,    # "moving-bus" mapped to "moving-other-vehicle" -----------mapped
              258: 2,    # "moving-truck"
              259: 2,}    # "moving-other-vehicle"}
    self.motion_remap = {
              0: 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
              1: 1,    # "car"
              2: 1,    # "bicycle"
              3: 1,    # "motorcycle"
              4: 1,    # "truck"
              5: 1,    # "on-rails" mapped to "other-vehicle" ---------------------mapped
              6: 1,    # "person"
              7: 1,    # "bicyclist"
              8: 1,    # "motorcyclist"
              9: 0,    # "road"
              10: 0,    # "parking"
              11: 0,  # "sidewalk"
              12: 0,  # "other-ground"
              13: 0,  # "building"
              14: 0,  # "fence"
              15: 0,  # "vegetation"
              16: 0,  # "trunk"
              17: 0,  # "terrain"
              18: 0,   # "pole"
              19: 0,   # "traffic-sign"
              20: 2,   # "moving-car"
              21: 2,  # "moving-bicyclist"
              22: 2,   # "moving-person"
              23: 2,  # "moving-motorcyclist"
              24: 2,  # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
              25: 2,   # "moving-truck"
                      }
    self.learning_remap = {
              0: 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
              1: 1,    # "car"
              2: 2,    # "bicycle"
              3: 3,    # "motorcycle"
              4: 4,    # "truck"
              5: 5,    # "on-rails" mapped to "other-vehicle" ---------------------mapped
              6: 6,    # "person"
              7: 7,    # "bicyclist"
              8: 8,    # "motorcyclist"
              9: 9,    # "road"
              10: 10,    # "parking"
              11: 11,  # "sidewalk"
              12: 12,  # "other-ground"
              13: 13,  # "building"
              14: 14,  # "fence"
              15: 15,  # "vegetation"
              16: 16,  # "trunk"
              17: 17,  # "terrain"
              18: 18,   # "pole"
              19: 19,   # "traffic-sign"
              20: 1,   # "moving-car"
              21: 7,  # "moving-bicyclist"
              22: 6,   # "moving-person"
              23: 8,  # "moving-motorcyclist"
              24: 5,  # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
              25: 4,   # "moving-truck"
      }

    if self.is_train:
        self.cache_size = 10000
    else:
        self.cache_size = 0

  def __get_bev__(self, index, verbose=False, check_sanity=False):
    # get bev from pc input
    # get item in tensor shape
    scan_file = self.scan_files[index]
    if verbose:
      print('checking scan file ', scan_file)
    if self.gt:
      label_file = self.label_files[index]

    # open a semantic laserscan
    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

    # open and obtain scan
    scan.open_scan(scan_file)
    if self.gt:
      scan.open_label(label_file)
      scan.sem_label      = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    pc = scan.points
    res, voxel_indices = voxelize_occupy(pc, voxel_size=self.voxel_size, extents=self.area_extents,
                                         use_feat_num_pts_in_pillar=False, return_indices=True)

    # Compile the batch of voxels, so that they can be fed into the network
    padded_voxel_points = res

    if self.gt:
      if self.label_dim == 2:
        dpt = gen_2d_grid_gt_from_pts(pc, scan.sem_label, point_remission=scan.remissions, grid_size=self.voxel_size[0:2], reordered=True, return_past_2d_disp_gt=False,
                     category_num=self.nclasses, extents=self.area_extents, frame_skip=self.future_frame_skip, return_num_pts_per_cell=True)
        pixel_category = np.argmax(dpt['voxel_cat_map'], axis=2)[:, :, np.newaxis]
      elif self.label_dim == 3:
        dpt = gen_3d_voxel_gt_from_pts(pc, scan.sem_label, point_remission=scan.remissions, grid_size=self.voxel_size[0:3], reordered=True, return_past_2d_disp_gt=False,
                     category_num=self.nclasses, extents=self.area_extents, frame_skip=self.future_frame_skip, return_num_pts_per_cell=True)
        pixel_category = dpt['voxel_cat_map'].astype(np.int8)

      dpt['padded_voxel_points'] = padded_voxel_points

      if verbose:
        for key, content in dpt.items():
          try:
            print(key, content.shape)
          except:
            print(key, content)
        for j in range(pixel_category.shape[2]):
          bev_label      = self.map(pixel_category[:, :, j], self.learning_map_inv)
          bev_label_color= scan.sem_color_lut[bev_label]
          # plot2d_img([bev_label], title_name=['layer {} pixel label'.format(j)], show_fig=True)
          plot2d_img([bev_label_color], title_name=['layer {} pixel label'.format(j)], show_fig=True)

      return dpt
    else:
      return padded_voxel_points

  def __get_rv__(self, index, extents=None, verbose=False):
    # get item in tensor shape
    scan_file = self.scan_files[index]
    if self.gt:
      label_file = self.label_files[index]

    # open a semantic laserscan
    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

    # open and obtain scan
    scan.open_scan(scan_file)
    if self.gt:
      scan.open_label(label_file)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    # make a tensor of the uncompressed data (with the max num points)
    unproj_points = scan.points

    if self.gt:
      unproj_labels = scan.sem_label
    else:
      unproj_labels = []

    if self.gt:
      proj_mask   = scan.proj_mask
      proj_labels = scan.proj_sem_label
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []

    proj_x    = scan.proj_x
    proj_y    = scan.proj_y

    return unproj_points, unproj_labels, proj_labels, proj_x, proj_y

  def __load_h5__(self, index, verbose=False):
    scan_file = self.scan_files[index]
    bev_file  = scan_file.replace('dataset/', 'preprocessed/')
    bev_file  = bev_file.replace('bin', 'h5')

    hf = h5py.File(bev_file, 'r')
    padded_voxel_points = hf.get('padded_voxel_points')[()]
    non_empty_map = hf.get('non_empty_map')[()]
    pixel_cat_map = hf.get('voxel_cat_map')[()]
    pixel_rem_map= hf.get('voxel_rem_map')[()]
    cell_pts_num = hf.get('cell_pts_num')[()]
    new_point_coord = hf.get('new_point_coord')[()]
    new_point_label = hf.get('new_point_label')[()]
    cell_pts_num    = cell_pts_num/np.max(cell_pts_num)
    if verbose:
      pixel_category = pixel_cat_map.astype(np.int8)
      for j in range(pixel_category.shape[2]):
        bev_label      = self.map(pixel_category[:, :, j], self.learning_map_inv)
        bev_label_color= scan.sem_color_lut[bev_label]
        plot2d_img([bev_label_color], title_name=['layer {} pixel label'.format(j)], show_fig=True)
        # plot2d_img([pixel_rem_map[:, :, j]], title_name=['layer {} pixel label'.format(j)], show_fig=True)

    return padded_voxel_points, cell_pts_num, pixel_rem_map, pixel_cat_map, non_empty_map, new_point_coord, new_point_label

  def gather_unproj(self, scan):
    unproj_n_points = scan.points.shape[0]
    unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
    unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
    if self.gt:
      unproj_labels = torch.full([self.max_points], 0, dtype=torch.int32)
      unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
    else:
      unproj_labels = []

    return unproj_n_points, unproj_xyz, unproj_range, unproj_remissions, unproj_labels

  def gather_proj(self, scan):
    unproj_n_points = scan.points.shape[0]
    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz   = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []

    proj_x = torch.full([self.max_points], 0, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
    proj_y = torch.full([self.max_points], 0, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

    return proj_range, proj_xyz, proj_remission, proj_mask, proj_labels, proj_x, proj_y

  def gather_proj_multi(self, scan):
    unique_indices_dict = {}
    # [1, 2, 4, 16, 32]
    for os, indices in scan.indices_dict.items():
      if os==1:
        curr_max_points = 4096 * 32
      else:
        curr_max_points = int(4096 * 32 / 2) # TODO
      num_indices = indices.shape[0]
      unique_indices = torch.full([curr_max_points], 0, dtype=torch.long)
      unique_indices[:num_indices] = torch.from_numpy(indices)
      unique_indices_dict[os] = unique_indices
      #
    return unique_indices_dict

  def getitem_on_the_fly(self, index, verbose=False):
    skip_num  = self.past_frame_skip
    scan_file = self.scan_files[index]
    index_frame = max(int(scan_file.split('.')[0].split('/')[-1]), self.num_sweep * skip_num) # 5
    index_seq   = int(scan_file.split('.')[0].split('/')[-3])
    path_seq    = scan_file.split('sequences')[0] + 'sequences/'
    padded_pts_input = [] # 
    padded_rem_input = [] # 
    padded_num_input = []

    if self.gt:
      label_file = self.label_files[index]
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

    # open and obtain scan by openning file
    scan.open_scan(scan_file)
    if self.augment and self.is_train:
        rot_ang = randint(-50, 50)/180 * np.pi
        rot_mat = np.array([[math.cos(rot_ang), -math.sin(rot_ang), 0],
                            [math.sin(rot_ang), math.cos(rot_ang), 0],
                            [0, 0, 1]])
        scan.points = np.dot(scan.points, rot_mat.T)
        decision = randint(0,1)
        if decision > 0:
            scan.points[:, 1] = - scan.points[:, 1]
        decision1 = randint(0,1)
        if decision1 > 0:
            scan.points[:, 0] = - scan.points[:, 0]

    # open label file
    if self.gt:
      scan.open_label(label_file)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)
      pc = scan.points
      if self.label_dim == 2:
        dpt = gen_2d_grid_gt_from_pts(pc, scan.sem_label, point_remission=scan.remissions, grid_size=self.voxel_size[0:2], reordered=True, return_past_2d_disp_gt=False,
                     category_num=self.nclasses, extents=self.area_extents, frame_skip=self.future_frame_skip, return_num_pts_per_cell=True)
      elif self.label_dim == 3:
        dpt = gen_3d_voxel_gt_from_pts(pc, scan.sem_label, point_remission=scan.remissions, grid_size=self.voxel_size[0:3], reordered=True, return_past_2d_disp_gt=False,
                     category_num=self.nclasses, extents=self.area_extents, frame_skip=self.future_frame_skip, return_num_pts_per_cell=True)

      pixel_cat_map   = dpt['voxel_cat_map']
      unproj_labels   = torch.full([self.max_points], 0, dtype=torch.int32)
      unproj_n_points = scan.points.shape[0]
      unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
      #
      if self.label_dim==2:
        pixel_cat_map = torch.from_numpy(np.argmax(dpt['voxel_cat_map'], axis=2).copy()) # now category map is [h, w], argmax due to soft prob
        viz_cat_map   = pixel_cat_map.numpy()
      else:
        pixel_cat_map = torch.from_numpy(pixel_cat_map.transpose(2, 0, 1).copy())
        viz_cat_map   = np.sum(pixel_cat_map.numpy(), axis=0)
      if verbose:
        plot_imgs([viz_cat_map], [f't'], title_name='gt', sub_name=index, save_fig=True, show_fig=False)
        plot_imgs([non_empty_map.squeeze().numpy()], [f't'], title_name='mask', sub_name=index, save_fig=True, show_fig=False)
    else:
      pixel_cat_map = []
      unproj_labels = []
      # unproj_n_points= 

    # append all related sequence points
    calibration = parse_calibration(path_seq + "{0:02d}/calib.txt".format(index_seq))
    poses = parse_poses(path_seq + "{0:02d}/poses.txt".format(index_seq), calibration)
    # pose
    pose = poses[index_frame]
    #pose 
    bev_list = []
    for k in range(0, self.num_sweep):
      past_pose = poses[index_frame - k*skip_num]
      scan_file = path_seq + '{0:02d}/velodyne/{1:06d}.bin'.format(index_seq, index_frame - k*skip_num)
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

      # open and obtain scan
      scan.open_scan(scan_file)
      diff = np.matmul(inv(pose), past_pose)
      points = np.ones((scan.points.shape[0], 4))
      points[:, 0:3] = scan.points[:, 0:3]
      scan.points = np.matmul(diff, points.T).T
      if self.augment and self.is_train:
        scan.points = np.dot(scan.points[:, :3], rot_mat.T)
        # random flip
        if decision > 0:
            scan.points[:, 1] = - scan.points[:, 1]
        if decision1 > 0:
            scan.points[:, 0] = - scan.points[:, 0]

      # by default it is 3d
      dpt = voxelize_occupy_detail(scan.points, point_remission=scan.remissions, grid_size=self.voxel_size[0:3], reordered=True, return_past_2d_disp_gt=False,
                       category_num=self.nclasses, extents=self.area_extents, frame_skip=self.future_frame_skip, return_num_pts_per_cell=True)
      if k==0:
        non_empty_map   = torch.from_numpy(dpt['non_empty_map'].copy()).unsqueeze(0)
        proj_n_points   = dpt['filter_idx'].shape[0]
        unproj_n_points = scan.points.shape[0]

        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)

        filter_idx      = torch.full([self.max_points], -1, dtype=torch.int32)
        filter_idx[:proj_n_points] = torch.from_numpy(dpt['filter_idx'])

        sorted_order    = torch.full([self.max_points], -1, dtype=torch.int32)
        sorted_order[:proj_n_points] = torch.from_numpy(dpt['sorted_order'])
        
        point_indices    = torch.full([self.max_points, 3], -1, dtype=torch.int32)
        sorted_order[:proj_n_points] = torch.from_numpy(dpt['point_indices'])

        if self.label_dim==3: # add another dimension
          non_empty_map   = non_empty_map.unsqueeze(0)

      padded_voxel_points = torch.from_numpy(dpt['voxels'].transpose(2, 0, 1).copy())
      pixel_rem_map       = torch.from_numpy(dpt['remissions'].transpose(2, 0, 1).copy())
      cell_pts_num        = torch.from_numpy(dpt['cell_pts_num'].transpose(2, 0, 1).copy())
      padded_pts_input.append(padded_voxel_points)
      padded_rem_input.append(pixel_rem_map)
      padded_num_input.append(cell_pts_num)
      bev_list.append(np.sum(dpt['voxels'], axis=2)) # for viz

    if verbose:
      plot_imgs(bev_list, [f'frame t-{k*skip_num}' for k in range(self.num_sweep)], title_name='Seq', sub_name=index, save_fig=True, grid_on=True, show_fig=False)

    if self.use_bev:  # (Batch, seq, z, h, w)
      input_per_frame = []
      for k in range(self.num_sweep):
        padded_input = torch.cat([padded_pts_input[k], padded_rem_input[k]], 0) # 28
        input_per_frame.append(padded_input)
      padded_input = torch.stack(input_per_frame, 0)
      return padded_input, non_empty_map, pixel_cat_map, point_indices, \
          unproj_xyz.transpose(1, 0), unproj_labels, filter_idx, sorted_order, unproj_n_points
    else:
      return []
    #   return unproj_xyz.transpose(1, 0), unproj_remissions, unproj_range, unproj_labels, \
    #             padded_input, non_empty_map, pixel_cat_map, point_coord,  \
    #                proj, proj_mask, proj_labels,  proj_x, proj_y, unique_indices1, unique_indices2, unique_indices4, unique_indices8, unique_indices16

  def preprocess_and_save(self, index, verbose=False):
    skip_num = 1
    self.num_sweep = 5 # manually change to 10 # TODO
    scan_file   = self.scan_files[index]
    index_seq   = int(scan_file.split('.')[0].split('/')[-3])
    index_frame = max(int(scan_file.split('.')[0].split('/')[-1]), self.num_sweep * skip_num) # 5
    path_seq    = scan_file.split('sequences')[0] + 'sequences/'
    directory   = self.save_preprocess_path + '/{0:02d}/'.format(index_seq)
    # if not os.path.exists(directory):
    #   os.makedirs(directory)
    save_file_name = self.save_preprocess_path + '/{0:02d}/{1:06d}.npy'.format(index_seq, index_frame)
    padded_pts_input = []
    padded_rem_input = []
    padded_num_input = []

    if self.gt:
      label_file = self.label_files[index]
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

    # open and obtain scan by openning file
    scan.open_scan(scan_file)
    # if self.augment and self.is_train:
    #     rot_ang = randint(-50, 50)/180 * np.pi
    #     rot_mat = np.array([[math.cos(rot_ang), -math.sin(rot_ang), 0],
    #                         [math.sin(rot_ang), math.cos(rot_ang), 0],
    #                         [0, 0, 1]])
    #     scan.points = np.dot(scan.points, rot_mat.T)
    #     decision = randint(0,1)
    #     if decision > 0:
    #         scan.points[:, 1] = - scan.points[:, 1]
    #     decision1 = randint(0,1)
    #     if decision1 > 0:
    #         scan.points[:, 0] = - scan.points[:, 0]

    # open label file
    save_data_dict = {}
    save_data_dict['num_past_pcs'] = self.num_sweep
    if self.gt:
      scan.open_label(label_file)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)
      pc = scan.points
      if self.label_dim == 2:
        dpt = gen_2d_grid_gt_from_pts(pc, scan.sem_label, point_remission=scan.remissions, grid_size=self.voxel_size[0:2], reordered=True, return_past_2d_disp_gt=False,
                     category_num=self.nclasses, extents=self.area_extents, frame_skip=self.future_frame_skip, return_num_pts_per_cell=True, return_indices_only=True)
      elif self.label_dim == 3:
        dpt = gen_3d_voxel_gt_from_pts(pc, scan.sem_label, point_remission=scan.remissions, grid_size=self.voxel_size[0:3], reordered=True, return_past_2d_disp_gt=False,
                     category_num=self.nclasses, extents=self.area_extents, frame_skip=self.future_frame_skip, return_num_pts_per_cell=True, return_indices_only=True)
        save_data_dict['num_divisions'] = dpt['num_divisions']
        save_data_dict['voxel_indices'] = dpt['voxel_indices']
        save_data_dict['rem_in_voxel'] = dpt['rem_in_voxel']
        save_data_dict['num_pts_in_voxel'] = dpt['num_points_in_voxel']
        save_data_dict['voxel_cat'] = dpt['voxel_cat']
    # append all related sequence points
    calibration = parse_calibration(path_seq + "{0:02d}/calib.txt".format(index_seq))
    poses = parse_poses(path_seq + "{0:02d}/poses.txt".format(index_seq), calibration)
    # pose
    pose = poses[index_frame]
    #
    bev_list = []
    for k in range(1, self.num_sweep):
      past_pose = poses[index_frame - k*skip_num]
      scan_file = path_seq + '{0:02d}/velodyne/{1:06d}.bin'.format(index_seq, index_frame - k*skip_num)
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

      # open and obtain scan
      scan.open_scan(scan_file)
      diff = np.matmul(inv(pose), past_pose)
      points = np.ones((scan.points.shape[0], 4))
      points[:, 0:3] = scan.points[:, 0:3]
      scan.points = np.matmul(diff, points.T).T

      # if self.augment and self.is_train:
      #   scan.points = np.dot(scan.points[:, :3], rot_mat.T)
      #   # random flip
      #   if decision > 0:
      #       scan.points[:, 1] = - scan.points[:, 1]
      #   if decision1 > 0:
      #       scan.points[:, 0] = - scan.points[:, 0]

      # by default it is 3d
      data_per_frame = voxelize_occupy_detail(scan.points, point_remission=scan.remissions, grid_size=self.voxel_size[0:3], reordered=True, return_past_2d_disp_gt=False,
                       category_num=self.nclasses, extents=self.area_extents, frame_skip=self.future_frame_skip, return_num_pts_per_cell=True, return_indices_only=True)
      save_data_dict[f'voxel_indices_{k}'] = data_per_frame['voxel_indices']
      save_data_dict[f'rem_in_voxel_{k}']   = data_per_frame['rem_in_voxel']
      save_data_dict[f'num_pts_in_voxel_{k}'] = data_per_frame['num_points_in_voxel']

    np.save(save_file_name, arr=save_data_dict)
    print('saving to ', save_file_name)

    return save_data_dict[f'voxel_indices']

  def __getitem__(self, idx, verbose=False):
    if self.compute_on_the_fly:
      data = self.getitem_on_the_fly(idx)

    skip_num = self.past_frame_skip
    if not self.gt: 
      non_empty_map = []
      pixel_cat_map = []
      return [], [], []
    # scan_file   = self.scan_files[idx]
    # index_frame = max(int(scan_file.split('.')[0].split('/')[-1]), self.num_sweep) # 5
    # index_seq   = int(scan_file.split('.')[0].split('/')[-3])
    padded_pts_input = []
    padded_rem_input = []
    padded_num_input = []

    if idx in self.cache:
      gt_dict = self.cache[idx]
    else: 
      gt_file_path    = self.indice_files[idx]
      # gt_file_path    = scan_file.split('sequences')[0] + f'preprocessed/{index_seq:02d}/{index_frame:06d}.npy'
      # gt_file_path    = scan_file.split('sequences')[0] + f'preprocessed/{index_seq}/{index_frame}.npy'
      # gt_file_path    = gt_file_path.replace('.bin', '.npy')
      gt_data_handle  = np.load(gt_file_path, allow_pickle=True)
      gt_dict         = gt_data_handle.item()
    # 
    num_divisions   = gt_dict['num_divisions']
    voxel_indices   = gt_dict['voxel_indices']
    voxel_cat       = gt_dict['voxel_cat']
    motion_state    = self.map(voxel_cat.astype(np.int8), self.motion_remap)
    voxel_cat       = self.map(voxel_cat.astype(np.int8), self.learning_remap)
    # if self.nclasses

    if len(self.cache) < self.cache_size:
      self.cache[idx] = gt_dict

    voxel_cat_map = np.zeros((num_divisions[0], num_divisions[1], num_divisions[2]), dtype=np.float32)
    voxel_cat_map[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = voxel_cat[:] # those with points will be assigned with the high-frequency label or soft label
    pixel_cat_map = torch.from_numpy(voxel_cat_map.transpose(2, 0, 1).copy())
    viz_cat_map   = np.sum(voxel_cat_map, axis=2)

    motion_state_map= np.zeros((num_divisions[0], num_divisions[1], num_divisions[2]), dtype=np.float32)
    motion_state_map[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]= motion_state[:]
    viz_mot_map   = np.sum(motion_state_map, axis=2)
    motion_state_map = torch.from_numpy(motion_state_map.transpose(2, 0, 1).copy())
    

    # Set the non-zero voxels to 1.0, which will be helpful for loss computation
    non_empty_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    non_empty_map[voxel_indices[:, 0], voxel_indices[:, 1]] = 1.0 # will take care of every non-empty voxels into consideration
    non_empty_map = torch.from_numpy(non_empty_map.copy()).unsqueeze(0).unsqueeze(1)

    # set 0 into 0
    motion_ignore_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    valid_indices = voxel_indices[np.where(motion_state>0)[0]]
    motion_ignore_map[valid_indices[:, 0], valid_indices[:, 1]] = 1.0 # will take care of every non-empty voxels into consideration
    motion_ignore_map = torch.from_numpy(motion_ignore_map.copy()).unsqueeze(0).unsqueeze(1)
    
    if verbose:
      plot_imgs([viz_cat_map], [f'fm_t'], title_name='gt', sub_name=idx, save_fig=True, show_fig=False)
      plot_imgs([viz_mot_map], [f'fm_t'], title_name='GT Motion', sub_name=idx, save_fig=True, show_fig=False)
      plot_imgs([non_empty_map.squeeze().numpy()], [f'mask_t'], title_name='cls', sub_name=idx, save_fig=True, show_fig=False)
      plot_imgs([motion_ignore_map.squeeze().numpy()], [f'mask_t'], title_name='motion', sub_name=idx, save_fig=True, show_fig=False)

    bev_list = []
    target_past_frames = [0] + list(np.sort(sample(range(1, 10), self.num_sweep - 1))) # TODO
    # target_past_frames = [0, 19]
    for i in target_past_frames:
      if i == 0:
        indices = voxel_indices 
        remissions = gt_dict['rem_in_voxel']
        num_pts    = gt_dict['num_pts_in_voxel']
      else:
        indices = gt_dict['voxel_indices_' + str(i)]
        remissions = gt_dict['rem_in_voxel_' + str(i)]
        num_pts    = gt_dict['num_pts_in_voxel_' + str(i)]
      curr_voxels = np.zeros(num_divisions, dtype=np.bool)
      curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

      voxel_rem_map = np.zeros((num_divisions[0], num_divisions[1], num_divisions[2]), dtype=np.float32)
      voxel_rem_map[indices[:, 0], indices[:, 1], indices[:, 2]] = remissions[:] # we combie remission according to pillars

      cell_pts_num = np.zeros((num_divisions[0], num_divisions[1], num_divisions[2]), dtype=np.float32)
      cell_pts_num[indices[:, 0], indices[:, 1], indices[:, 2]]  = num_pts[:]

      padded_voxel_points = torch.from_numpy(curr_voxels.transpose(2, 0, 1).copy().astype(np.float32))
      pixel_rem_map       = torch.from_numpy(voxel_rem_map.transpose(2, 0, 1).copy())
      cell_pts_num        = torch.from_numpy(cell_pts_num.transpose(2, 0, 1).copy())

      padded_pts_input.append(padded_voxel_points)
      padded_rem_input.append(pixel_rem_map)
      padded_num_input.append(cell_pts_num)

      bev_list.append(np.sum(curr_voxels, axis=2)) # for viz

    if verbose:
      plot_imgs(bev_list, [f'frame\t t-{k}' for k in target_past_frames], title_name='Input', sub_name=index, save_fig=True, show_fig=False)
      plot_imgs([np.sum(np.array(bev_list), axis=0)], [f'{self.num_sweep}'], title_name='Overlap\tSeq', sub_name=index, save_fig=True, show_fig=False)
      plot_imgs([abs(bev_list[0] - bev_list[1])], [f'{self.num_sweep}'], title_name='Substract\tSeq', sub_name=index, save_fig=True, show_fig=False)

    input_per_frame = []
    for k in range(self.num_sweep):
      padded_input = torch.cat([padded_pts_input[k], padded_rem_input[k]], 0) # 28
      input_per_frame.append(padded_input)

    padded_input = torch.stack(input_per_frame, 0)

    return padded_input, non_empty_map, pixel_cat_map, motion_ignore_map, motion_state_map  #

  def __len__(self):
    return len(self.scan_files)

  def __len_valid__(self):
    return len(self.indice_files)


  def sanity_check(self, pc_label, proj_label, point_coord, filter_idx=None, class_ignore=None, class_strings=None, class_inv_remap=None, target='bev'):
    """
    we only need to get the point-to-pixel mapping index
    P[i] in P[x, y], so proj_label(P[i]) = proj_labels[x, y]
    pc: [N, 3]
    pc_label: [N]
    proj_labels: [W, L, H] or [W, L]
    """
    proj_x = point_coord[:, 0]
    proj_y = point_coord[:, 1]
    if point_coord.shape[1]>2:
      proj_z = point_coord[:, 2]
      unproj_argmax = proj_label[proj_x, proj_y, proj_z]
    else:
      unproj_argmax = proj_label[proj_x, proj_y]
    if filter_idx is not None:
      pc_label = pc_label[filter_idx]
      unproj_argmax = unproj_argmax[filter_idx]

    # check overall accuracy & miou by building
    nr_classes = self.nclasses
    ignore     = []
    if class_ignore is not None:
      for cl, ign in class_ignore.items():
        if ign:
          x_cl = int(cl)
          ignore.append(x_cl)
          print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

    # create evaluator
    device    = torch.device("cpu")
    evaluator = iouEval(nr_classes, device, ignore)
    evaluator.reset()
    # add single scan to evaluation
    evaluator.addBatch(unproj_argmax, pc_label)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    print('Validation set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard))
    # print also classwise
    if class_strings and class_inv_remap:
      for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
          print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
              i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

    return m_jaccard, m_accuracy


  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               cfg,               # training cfg
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True):  # shuffle training set?
    super(Parser, self).__init__()

    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.cfg = cfg
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = min(len(self.learning_map_inv), 20)
    if cfg.DATASET.augment:
        print('we are using data augmentation!!')
    # Data loading code
    self.train_dataset = SemanticKitti(root=self.root,
                                       sequences=self.train_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       cfg=self.cfg,
                                       max_points=max_points,
                                       gt=self.gt,
                                       )
    self.train_sampler = None
    if cfg.TRAIN.distributed:
      print('using training sampler with worlde size, and rank ', cfg.TRAIN.rank, cfg.TRAIN.world_size)
      self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,
                                                                    num_replicas=cfg.TRAIN.world_size,
                                                                    rank=cfg.TRAIN.rank)
    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=(self.shuffle_train and self.train_sampler is None) ,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   sampler=self.train_sampler,
                                                   drop_last=True)
    assert len(self.trainloader) > 0
    self.trainiter = iter(self.trainloader)

    self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       cfg=self.cfg,
                                       max_points=max_points,
                                       gt=self.gt,
                                       is_train=False)
    # self.valid_sampler = None
    # if cfg.TRAIN.distributed:
    #   print('using training sampler')
    #   self.valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset,
    #                                                                 num_replicas=cfg.TRAIN.world_size,
    #                                                                 rank=cfg.TRAIN.rank)
    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)

    if self.test_sequences:
      self.test_dataset = SemanticKitti(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        cfg=self.cfg,
                                        max_points=max_points,
                                        gt=False,
                                        is_train=False)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_motion(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.motion_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)

def breakpoint():
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    # all config args
    quick_test   = True
    is_verbose   = True
    save_flag    = False
    check_sanity = False
    check_input  = True
    check_bev    = False
    # base_data = '/homes/xili/datasets'
    # base_code = '/homes/xili/cv0_home/AutoSeg/fuse_1.1d'
    base_data    = '/groups/CESCA-CV/dataset'
    base_code    = '/home/lxiaol9/4DAutoSeg/fuse_1d'

    print('testing on the bev auto generation!')
    root = base_data
    DATA = yaml.safe_load(open(base_code + '/config/labels/semantic-kitti-all.yaml', 'r'))
    cfg.merge_from_file(base_code + '/config/arch/2_kitti-bev-motionnet.yaml')
    train_sequences= DATA["split"]["train"]
    test_sequences = DATA["split"]["test"]
    val_sequences  = DATA["split"]["valid"]

    #
    labels=DATA["labels"]
    color_map=DATA["color_map"]
    learning_map=DATA["learning_map"]
    learning_map_inv=DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]

    max_points=cfg.DATASET.max_points
    gt=True

    # Data loading code
    print('root is ', root)
    train_dataset = SemanticKitti(root=root,
                                       sequences=val_sequences,
                                       labels=labels,
                                       color_map=color_map,
                                       learning_map=learning_map,
                                       learning_map_inv=learning_map_inv,
                                       cfg=cfg,
                                       past_frame_skip=2,
                                       max_points=max_points,
                                       gt=gt)
    train_dataset.use_rv= True
    train_dataset.use_bev= True
    cover_ratios = []


    if quick_test:
      check_list = np.sort(sample(range(1,train_dataset.__len_valid__()), 10))
    else:
      check_list = range(0, train_dataset.__len__())

    start_t = time.time()
    for index in check_list:
      # print('loading the bev data with index ', index)
      # dp = train_dataset.__get_bev__(index, verbose=is_verbose)
      # print(list(dp.keys()))
      # cover_ratios.append(dp['cover_ratio'])
      if check_sanity:
        pc_label    = dp['new_point_label']
        proj_label  = dp['pixel_cat_map']
        point_coord = dp['new_point_coord']

        miou, acc   = train_dataset.sanity_check(pc_label, proj_label, point_coord, class_ignore=class_ignore, class_strings=labels, class_inv_remap=learning_map_inv, target='bev')
        print('The upper limit is miou: {} acc: {},  discretization under grid_size: {}, extents: {}'.format(miou, acc, train_dataset.voxel_size, train_dataset.area_extents))

      if check_input:
        # p = train_dataset.preprocess_and_save(index, verbose=False)
        p = train_dataset.__getitem__(index, verbose=is_verbose)
        for j in range(len(p)):
          print(p[j].shape)

    # print('average ratio is ', sum(cover_ratios)/len(cover_ratios))
    print('loading takes {} seconds on average'.format( (time.time() - start_t)/len(check_list)))
    #
