import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
from os import makedirs, remove
from os.path import exists, join
from tqdm import tqdm
import igl

osp = os.path
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# import hydra
# import scipy
# import scipy.io
# from hydra import utils
# from omegaconf import DictConfig, ListConfig, OmegaConf
# from multiprocessing import Manager
# from pytransform3d.rotations import *
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import _init_paths
from utilities.dataset import ContactPose, get_object_names
import utilities.misc as mutils
import utilities.rendering as rutils
from utilities.rendering import DepthRenderer
from common.vis_utils import plot3d_pts, plot_arrows, plot_imgs, visualize_mesh, visualize_pointcloud
from global_info import global_info

infos           = global_info()
base_path       = infos.base_path
group_dir       = infos.group_path
second_path     = infos.second_path
grasps_meta     = infos.grasps_meta

# may need to augment the data
def calculate_3d_backprojections(depth, K, height=480, width=640, inds=None, verbose=False):
    # backproject to camera space
    xmap = np.array([[j for i in range(width)] for j in range(height)])
    ymap = np.array([[i for i in range(width)] for j in range(height)])
    cam_cx = K[0, 2]
    cam_cy = K[1, 2]
    cam_fx = K[0, 0]
    cam_fy = K[1, 1]
    # cam_fx = K[0, 0]
    # cam_fy = K[1, 1]
    cam_scale = 1
    pt2 = depth / cam_scale
    pt0 = (ymap - cam_cx) * pt2 / cam_fx
    pt1 = (xmap - cam_cy) * pt2 / cam_fy
    cloud = np.stack((pt0, pt1, pt2), axis=-1)
    # if inds is not None:
    #     cloud = cloud[inds[0], inds[1]]
    if verbose:
        plot3d_pts([[cloud.reshape(-1, 3)]], [['Part {}'.format(j) for j in range(1)]], s=3**2, title_name=['cam pc'], sub_name=str(0), axis_off=False, save_fig=False)

    return cloud

def show_rendering_output(renderers, color_im, camera_name, frame_idx, crop_size=-1):
    joints = cp.projected_hand_joints(camera_name, frame_idx)
    if crop_size > 0:
        color_im, _ = mutils.crop_image(color_im, joints, crop_size)

    # object rendering
    object_rendering = renderers['object'].render(cp.object_pose(camera_name, frame_idx))
    if crop_size > 0:
        object_rendering, _ = mutils.crop_image(object_rendering, joints, crop_size)
    object_mask = object_rendering > 0
    color_im[object_mask] = (0, 255, 255)  # yellow

    # hand rendering
    both_hands_rendering = []
    for renderer, mask_color in zip(renderers['hands'], ((0, 255, 0), (0, 0, 255))):
        if renderer is None:  # this hand is not present for this grasp
            continue
        # hand meshes are already in the object coordinate system, so we can use
        # object pose for rendering
        rendering = renderer.render(cp.object_pose(camera_name, frame_idx))
        if crop_size > 0:
            rendering, _ = mutils.crop_image(rendering, joints, crop_size)
        both_hands_rendering.append(rendering)
        mask = rendering > 0
        color_im[mask] = mask_color
    both_hands_rendering = np.dstack(both_hands_rendering).max(2)

    # show
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(color_im[:, :, ::-1])
    ax0.set_title('Masks')
    ax1.imshow(object_rendering)
    ax1.set_title('Object Depth')
    ax2.imshow(both_hands_rendering)
    ax2.set_title('Hand Depth')
    fig.suptitle(camera_name)

#Dataset
class BaseDataset():
    def __init__(self, cfg=None, mode='train', domain=None, add_noise=False, fixed_order=False, is_testing=False):
        self.cps = []
        self.basenames = []
        subjects = [28]
        activities = ['use']
        objects = ['bowl']
        self.n_data = 0

        #
        self.cfg = cfg
        # basic properties
        # self.batch_size   = cfg.batch_size
        self.mode         = mode
        self.domain       = domain
        self.add_noise    = add_noise
        self.fixed_order  = fixed_order

        # loop all objects
        for subject in subjects:
            for act in activities:
                for obj in objects:
                    self.cps.append(ContactPose(subject, act, obj, root_dir='/home/dragon/Dropbox/ICML2021/code/ContactPose/data'))
                    self.basenames.append(f'{subject}_{act}_{obj}')
                    self.n_data += self.cps[-1]._n_frames

    def __len__(self):
        """
        Return the length of data here
        """
        return self.n_data

    def get_sample_pair(self, idx, debug=False):
        p_arr  = None
        gt_points  = None
        instance_name  = None
        instance_name1 = None
        return {"raw": p_arr, "real": gt_points, "raw_id": instance_name, "real_id": instance_name1}

    def get_sample_mine(self, idx, debug=False):
        gt_points  = None
        instance_name  = None
        return {"points": gt_points, "id": instance_name}

    def __getitem__(self, idx):
        try:
            if self.task == 'pcloud_completion':
                sample = self.get_sample_mine(idx)
            else:
                idx = self.all_ids[idx]
                sample = self.get_sample_pair(idx)
        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self)-1)
            idx = self.all_ids[random_idx]
            if self.task == 'pcloud_completion':
                sample = self.get_sample_mine(idx)
            else:
                sample = self.get_sample_pair(idx)

        return sample

def create_renderers(camera_name, root_dir='data'):
    # renderer for object mesh
    # note the mesh_scale, ContactPose object models are in units of mm
    object_renderer = rutils.DepthRenderer(object_name, cp.K(camera_name), camera_name, mesh_scale=1e-3, root_dir=root_dir)

    # hand renderers
    hand_renderers = []
    for mesh in cp.mano_meshes():
        if mesh is None:  # this hand is not present for this grasp
            hand_renderers.append(None)
        else:
            renderer = rutils.DepthRenderer(mesh, cp.K(camera_name), camera_name)
            hand_renderers.append(renderer)
    return {'object': object_renderer, 'hands': hand_renderers}

if __name__ == '__main__':
    # number of subjects, activity, object
    all_data = BaseDataset()
    root_dir = '/home/dragon/Dropbox/ICML2021/code/ContactPose/data'
    s_ind = 0
    f_ind = 7
    c_ind = 0
    verbose = True
    viz_m = False # mask
    viz_h = True # hand
    viz_raw = True # raw depth/rgb data
    viz_p = True
    cal_m = False

    # start fetching data
    cp = all_data.cps[s_ind]
    p_num, intent, object_name = all_data.basenames[s_ind].split('_')
    p_num = int(p_num)
    all_cameras = cp.valid_cameras # 'kinect2_left', 'kinect2_middle', 'kinect2_right'
    # read full images
    frame_idx = f_ind
    color_im_filenames = cp.image_filenames('color', frame_idx)
    depth_im_filenames = cp.image_filenames('depth', frame_idx)
    print(color_im_filenames)
    color_ims = {n: cv2.imread(f, -1) for n,f in color_im_filenames.items()}
    depth_ims = {n: cv2.imread(f, -1) for n,f in depth_im_filenames.items()}

    # Full images
    camera_name = all_cameras[c_ind]
    color_im = color_ims[camera_name]
    depth_im = depth_ims[camera_name]

    # # camera intrinsics
    # for camera_name in cp.valid_cameras:
    #     print('##### Camera: {:s}'.format(camera_name))
    #     print(cp.K(camera_name))
    #
    # # object pose w.r.t. camera
    # for camera_name in cp.valid_cameras:
    #     print('##### Camera: {:s}'.format(camera_name))
    #     print(cp.object_pose(camera_name, frame_idx))
    # 0 mask
    # foreground mask
    K = cp.K(camera_name)
    A = mutils.get_A(camera_name)
    print('A: ', A)
    cTo = cp.object_pose(camera_name, frame_idx)
    print('cTo: ', cTo)

    # 1. backproject
    print('depth_im: ', depth_im.shape, np.min(depth_im), np.max(depth_im))
    # pts_cam = calculate_3d_backprojections(depth_im/1000, cp.K(camera_name), height=cp.im_size(camera_name)[1], width=cp.im_size(camera_name)[0], inds=None, verbose=True)
    # # if verbose and viz_p:
    # #     visualize_pointcloud([dp[0], dp[1]], title_name='partial + complete', backend='pyrender')
    #
    # # 2. pose inverse transformed
    # pts_canon = np.matmul(pts, cTo)

    # get projected hand joints and object markers
    joints = cp.projected_hand_joints(camera_name, frame_idx)
    markers = cp.projected_object_markers(camera_name, frame_idx)

    # draw on image
    if verbose and viz_raw:
        print('drawing ')
        color_im = mutils.draw_hands(color_im, joints)
        color_im = mutils.draw_object_markers(color_im, markers)
        depth_im = mutils.draw_hands(depth_im, joints)
        depth_im = mutils.draw_object_markers(depth_im, markers)
        print('vizing')
        plt.figure()
        plt.title(camera_name)
        plt.imshow(np.hstack((color_im, depth_im))[:, :, ::-1])  # BGR -> RGB
        plt.show()

    # frame_idx = np.random.choice(len(cp))
    crop_size = 400
    plt.close('all')
    print('---start for mask rendering')
    for camera_name in ('kinect2_left', 'kinect2_right', 'kinect2_middle'):
        color_im = cv2.imread(cp.image_filenames('color', frame_idx)[camera_name])
        renderers = create_renderers(camera_name, root_dir=root_dir)
        show_rendering_output(renderers, color_im, camera_name, frame_idx, crop_size)
    print('---start for mask showing')
    plt.show()

    if cal_m:
        do_rgb=True
        do_depth=True
        do_grabcut=True,
        depth_percentile_thresh=30
        mask_dilation=5

        # # collect mask,
        crop_size = 400

        K = cp.K(camera_name)
        A = mutils.get_A(camera_name)
        print('camera intrinsics: \n', K)
        print('camera affine: \n', A)
        renderer = DepthRenderer(object_name, K, camera_name, 1e-3, root_dir=root_dir)
        output_dir = osp.join(cp.data_dir, 'images', camera_name)
        for d in ('color', 'depth', 'projections'):
          dd = osp.join(output_dir, d)
          if not osp.isdir(dd):
            os.makedirs(dd)

        joints = cp.projected_hand_joints(camera_name, frame_idx)

        # crop images
        rgb_im, _ = mutils.crop_image(color_im, joints, crop_size)
        depth_im, crop_tl = mutils.crop_image(depth_im, joints, crop_size)
        this_A = np.copy(A) # A
        A = np.asarray([[1, 0, -crop_tl[0]], [0, 1, -crop_tl[1]], [0, 0, 1]]) @ A
        cTo = cp.object_pose(camera_name, frame_idx)
        P = this_A @ K @ cTo[:3] # object pose, intrinsics, A camera affine matrix, z

        if verbose and viz_m:
            plot_imgs([rgb_im, depth_im, mask], ['rgb', 'depth', 'mask'])

    # contact infos
    print(cp.contactmap_filename)

    # MANO parameters - pose PCA components 'pose', shape PCA components 'betas', root transform 'hTm'
    for hand_name, p in zip(('Left', 'Right'), cp.mano_params):
        print('##### Hand: {:s}'.format(hand_name))
        if p is None:
            print('Absent')
        else:
            print('pose: size {:d}'.format(len(p['pose'])))
            print('betas: size {:d}'.format(len(p['betas'])))
            print('hTm: ')
            print(p['hTm'])

    # MANO meshes - vertices, face indices, and 21 joints
    for hand_name, mesh in zip(('Left', 'Right'), cp.mano_meshes(frame_idx=frame_idx)):
        if mesh is None:
            print('empty hand')
            continue
        print('### Hand {:s}'.format(hand_name))
        print('vertices: ', mesh['vertices'].shape)
        print('face indices: ', mesh['faces'].shape)
        print('joints: ', mesh['joints'].shape)

        # load objs
        obj_vert, f1 = igl.read_triangle_mesh(cp.contactmap_filename)
        if verbose and viz_h:
            plot3d_pts([[mesh['vertices'], obj_vert]], [[hand_name, 'obj']])
            # visualize_mesh(mesh, mode='mesh', backend='matplotlib', title_name='default')

    # hand joints infos
    for hand_name, j in zip(('Left', 'Right'), cp.hand_joints(frame_idx=frame_idx)):
        print('##### Hand: {:s}'.format(hand_name))
        if j is not None:
            print(j.shape)
        else:
            print('Absent')
# PyQt5 5.11.3
#
