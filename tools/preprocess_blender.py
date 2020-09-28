import matplotlib

import matplotlib.pyplot as plt

import os
import sys
import time
import random as rdn
import multiprocessing
from multiprocessing import Process
import h5py
import yaml
import json
import copy
import collections
import argparse
import scipy.io
import math
import numpy as np
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from skimage.color import rgb2gray
from pytransform3d.rotations import *

from collections import OrderedDict
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
# import OpenEXR as exr
from oiio import OpenImageIO as oiio

import Imath

# custom libs
import _init_paths
from common.transformations import euler_matrix, quaternion_matrix
from common.vis_utils import plot3d_pts
from common.data_utils import collect_file, split_dataset
from global_info import global_info

infos           = global_info()
grasps_meta     = infos.grasps_meta
mano_path       = infos.mano_path
whole_obj       = infos.whole_obj
hand_mesh       = infos.hand_mesh


def breakpoint():
    import pdb;pdb.set_trace()
# backproject pixels into 3D points

def color_srgb_to_linear_backward_array(c):
    # c must be 3D ARRARY
    vc = np.zeros_like(c)
    inds = np.where(c<0.0031308)
    vc[inds[0], inds[1], inds[2]] = c[inds[0], inds[1], inds[2]] * 12.92
    inds = np.where(c>=0.0031308)
    vc[inds[0], inds[1], inds[2]] = c[inds[0], inds[1], inds[2]] ** (1/2.4) * (1.055) - 0.055

    return vc

def color_srgb_to_linear_backward(c):
    if c < 0.0031308:
        return c * 12.92
    else:
        return c ** (1/2.4) * (1.055) - 0.055


def readEXR(filename):
    """Read color + depth data from EXR image file.

    Parameters
    ----------
    filename : str
        File path.

    Returns
    -------
    img : RGB or RGBA image in float32 format. Each color channel
          lies within the interval [0, 1].
          Color conversion from linear RGB to standard RGB is performed
          internally. See https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
          for more information.

    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
    """

    # exrfile = exr.InputFile(filename)
    # header = exrfile.header()
    # # breakpoint()
    # dw = header['dataWindow']
    # isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    # channelData = dict()

    # # convert all channels in the image to numpy arrays
    # for c in header['channels']:
    #     C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
    #     C = Image.frombytes("F", isize, C)
    #     C = np.reshape(C, isize)

    #     channelData[c] = C
    # colorChannels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
    # img = np.concatenate([channelData[c][...,np.newaxis] for c in colorChannels], axis=2)
    # # print(img[np.where(img<np.min(img)+2)[0], np.where(img<np.min(img)+2)[1], np.where(img<np.min(img)+2)[2]])
    # # # linear to standard RGB
    # # img[..., :3] = np.where(img[..., :3] <= 0.0031308,
    # #                         12.92 * img[..., :3],
    # #                         1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)

    # # sanitize image to be in range [0, 1]
    # # img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))

    # Z = None if 'Z' not in header['channels'] else channelData['Z']

    inbuf=oiio.ImageInput.open(filename)
    img  = inbuf.read_image()
    Z = None
    inbuf.close()


    return img, Z

class PoseDataset():
    def __init__(self, root_dir, render_path, item, second_dir=None, num_points=1024,  objs=[], add_noise=False, noise_trans=0, mode='train', refine=False, selected_list=None, is_debug=False):
        """
        num is the number of points chosen feeding into PointNet
        """
        self.is_debug = is_debug
        self.mode     = mode
        self.max_lnk  = 10
        self.root_dir  = root_dir
        if second_dir is None:
            self.second_dir = root_dir
        else:
            self.second_dir = second_dir
        self.dataset_render = render_path
        self.models_dir  = root_dir + '/objects'
        self.objnamelist = os.listdir(self.dataset_render)
        self.mode        = mode
        self.list_rgb    = []
        self.list_depth  = []
        self.list_label  = []
        self.list_mask   = []
        self.list_meta   = []
        self.list_obj    = []
        self.list_instance = []
        # obj/instance/arti/grasp_ind

        self.list_grasp = []
        self.list_status = []
        self.list_rank = []
        self.meta_dict = {}
        self.urdf_dict = {}
        self.pt_dict   = {}
        self.noise_trans = noise_trans
        self.refine   = refine

        ins_count = 0
        if self.is_debug:
            from mpl_toolkits.mplot3d import Axes3D
        obj_category      = item
        meta_dict_obj     = {}
        urdf_dict_obj     = {}

        base_path = self.dataset_render + '/' + obj_category

        meta     = {} # instance, arti_ind, grasp_ind
        group_dirs = os.listdir(base_path)
        group_dirs.sort()

        # ********* ins_arti_grasp or could be only folders containing  ************ #
        for group_dir in group_dirs:
            attrs     = group_dir.split('_')
            ins       = attrs[0]
            art_index = attrs[1]
            grasp_ind = attrs[2]
            if selected_list is not None and ins not in selected_list:
                continue

            sub_dir = base_path + '/' + group_dir
            for view_ind in range(len(os.listdir(sub_dir + '/rgba'))):
                self.list_rgb.append(sub_dir + '/rgba/{:04d}.png'.format(view_ind))
                self.list_depth.append(sub_dir + '/depth/{:04d}.png0001.exr'.format(view_ind))
                self.list_mask.append(sub_dir + '/mask/{:04d}.png'.format(view_ind))
                self.list_label.append(sub_dir + '/label/{:04d}.png'.format(view_ind)) # nocs value
                self.list_meta.append(sub_dir + '/meta_{:04d}.mat'.format(view_ind))
                self.list_obj.append(obj_category)
                self.list_instance.append(ins)
                self.list_status.append(art_index)
                self.list_grasp.append(grasp_ind)
                self.list_rank.append(int(view_ind))

            if ins not in urdf_dict_obj:
                urdf_ins = {}
                tree_urdf     = ET.parse(self.root_dir + "/urdf/" + obj_category + '/' + ins + "/syn.urdf") # todo
                root_urdf     = tree_urdf.getroot()
                rpy_xyz       = {}
                list_xyz      = [None] * self.max_lnk
                list_rpy      = [None] * self.max_lnk
                list_box      = [None] * self.max_lnk
                # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
                num_links     = 0
                for link in root_urdf.iter('link'):
                    num_links += 1
                    index_link = None
                    if link.attrib['name']=='base_link':
                        index_link = 0
                    else:
                        index_link = int(link.attrib['name'])
                    for visual in link.iter('visual'):
                        for origin in visual.iter('origin'):
                            list_xyz[index_link] = [float(x) for x in origin.attrib['xyz'].split()]
                            list_rpy[index_link] = [float(x) for x in origin.attrib['rpy'].split()]

                rpy_xyz['xyz']   = list_xyz
                rpy_xyz['rpy']   = list_rpy
                urdf_ins['link'] = rpy_xyz

                rpy_xyz       = {}
                list_xyz      = [None] * self.max_lnk
                list_rpy      = [None] * self.max_lnk
                list_axis     = [None] * self.max_lnk
                # here we still have to read the URDF file
                for joint in root_urdf.iter('joint'):
                    index_joint = int(joint.attrib['name'][0])
                    for origin in joint.iter('origin'):
                        list_xyz[index_joint] = [float(x) for x in origin.attrib['xyz'].split()]
                        list_rpy[index_joint] = [float(x) for x in origin.attrib['rpy'].split()]
                    for axis in joint.iter('axis'):
                        list_axis[index_joint]= [float(x) for x in axis.attrib['xyz'].split()]
                rpy_xyz['xyz']       = list_xyz
                rpy_xyz['rpy']       = list_rpy
                rpy_xyz['axis']      = list_axis

                urdf_ins['joint']    = rpy_xyz
                urdf_ins['num_links']= num_links

                # meta_dict_obj[ins]  = meta
                urdf_dict_obj[ins]  = urdf_ins
                print("Object {} instance {} buffer loaded".format(obj_category, ins))

        # self.meta_dict[obj_category] = meta_dict_obj
        self.urdf_dict[obj_category] = urdf_dict_obj
        self.joints_dict = np.load(f'{grasps_meta}/{obj_category}_joints.npy', allow_pickle=True).item()
        self.contacts_dict=np.load(f'{grasps_meta}/{obj_category }_contacts.npy', allow_pickle=True).item()
        self.length = len(self.list_rgb)
        self.height = 480
        self.width  = 640
        self.xmap = np.array([[j for i in range(self.width)] for j in range(self.height)])
        self.ymap = np.array([[i for i in range(self.width)] for j in range(self.height)])

        self.num    = num_points
        self.add_noise = add_noise

    def __len__(self):
        return self.length

    def backproject(self, depth, C, P, RT, K):
        """
        C is camera.location
        """
        # compute projection matrix
        # P, RT, K = self.compute_projection_matrix()
        P = np.matrix(P)
        Pinv = np.linalg.pinv(P)

        # compute the 3D points
        width = depth.shape[1]
        height = depth.shape[0]
        points = np.zeros((height, width, 3), dtype=np.float32)
        world_points = np.zeros((height, width, 3), dtype=np.float32)

        # camera location
        C = np.matrix(C).transpose()
        Cmat = np.tile(C, (1, width*height))

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

        # backproject to camera space
        xmap = self.xmap
        ymap = self.ymap
        cam_cx = K[0, 2]
        cam_cy = K[1, 2]
        cam_fx = K[0, 0]
        cam_fy = K[0, 0]
        # cam_fx = K[0, 0]
        # cam_fy = K[1, 1]
        cam_scale = 1
        pt2 = depth / cam_scale
        pt0 = (ymap - cam_cx) * pt2 / cam_fx
        pt1 = (xmap - cam_cy) * pt2 / cam_fy
        cloud = np.stack((pt0, pt1, pt2), axis=-1)

        # cloud = np.dot(np.linalg.pinv(K), x2d.T).reshape(height, width, 3)

        # breakpoint()
        x3d = Pinv * x2d.transpose()
        x3d[0,:] = x3d[0,:] / x3d[3,:]
        x3d[1,:] = x3d[1,:] / x3d[3,:]
        x3d[2,:] = x3d[2,:] / x3d[3,:]
        x3d = x3d[:3,:]

        # compute the ray
        R = x3d - Cmat

        # compute the norm
        N = np.linalg.norm(R, axis=0)

        # normalization
        R = np.divide(R, np.tile(N, (3,1)))

        # compute the 3D points
        X = Cmat + np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
        world_points[y, x, 0] = X[0,:].reshape(height, width)
        world_points[y, x, 1] = X[2,:].reshape(height, width)
        world_points[y, x, 2] = X[1,:].reshape(height, width)
        points = cloud
        if 0:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            perm = np.random.permutation(np.arange(height*width))
            index = perm[:10000]
            X = points[:,:,0].flatten()
            Y = points[:,:,1].flatten()
            Z = points[:,:,2].flatten()
            ax.scatter(X[index], Y[index], Z[index], c='r', marker='o')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # ax.set_aspect('equal')
            plt.show()

        return cloud, world_points

    # sspre
    def __preprocess_and_save__(self, index):
        obj_category       = self.list_obj[index]
        ins                = self.list_instance[index]
        arti_ind           = self.list_status[index]
        grasp_ind          = self.list_grasp[index]
        frame_order        = self.list_rank[index]
        print('ins {}, arti_ind {}, grasp_ind {}, frame_order {}'.format(ins, arti_ind, grasp_ind, frame_order))

        # collect joints & contacts
        joints = self.joints_dict[f'{ins}_{arti_ind}_{grasp_ind}']
        # joints[:, 1], joints[:, 2] = joints[:, 2], joints[:, 1]
        contacts= self.contacts_dict[f'{ins}_{arti_ind}_{grasp_ind}']

        #>>>>>>>>>>> core code
        alpha = -np.pi/2
        correct_R = matrix_from_euler_xyz([alpha, 0, 0])

        # transfrom joints into blender coords
        joints = np.dot(correct_R, joints.T).T
        contacts = np.dot(correct_R, contacts.T).T

        # use second_dir to seave hdf5
        h5_save_path       = self.second_dir + '/hdf5/' + obj_category + '/' + ins + '_' + arti_ind + '_' + grasp_ind
        if (not os.path.exists(h5_save_path)):
            os.makedirs(h5_save_path)
        h5_save_name       = h5_save_path  + '/{}.h5'.format(frame_order)

        # if os.path.exists(h5_save_name):
        #     print('skipping ')
        #     return None

        num_parts          = self.urdf_dict[obj_category][ins]['num_links'] + 1 # add hand path

        # model_offsets      = self.urdf_dict[obj_category][ins]['link']
        # joint_offsets      = self.urdf_dict[obj_category][ins]['joint']

        parts_model_point  = [None]*num_parts
        parts_world_point  = [None]*num_parts
        parts_target_point = [None]*num_parts

        parts_cloud_cam    = [None]*num_parts
        parts_cloud_world  = [None]*num_parts
        parts_cloud_canon  = [None]*num_parts
        parts_cloud_urdf   = [None]*num_parts
        parts_cloud_norm   = [None]*num_parts

        parts_world_pos    = [None]*num_parts
        parts_world_orn    = [None]*num_parts
        parts_urdf_pos     = [None]*num_parts
        parts_urdf_orn     = [None]*num_parts
        parts_urdf_box     = [None]*num_parts

        parts_model2world  = [None]*num_parts
        parts_canon2urdf   = [None]*num_parts
        parts_target_r     = [None]*num_parts
        parts_target_t     = [None]*num_parts

        parts_mask         = [None]*num_parts
        choose_x           = [None]*num_parts
        choose_y           = [None]*num_parts
        choose_to_whole    = [None]*num_parts

        # rgb/depth/label
        img   = np.array(Image.open(self.list_rgb[index]))

        depth, Z = readEXR(self.list_depth[index])
        # print(depth[np.where(depth<np.min(depth)+2)[0], np.where(depth<np.min(depth)+2)[1], np.where(depth<np.min(depth)+2)[2]])
        depth[np.where(depth>254)] = 0

        label = np.array(Image.open(self.list_label[index]))/255.0 #
        label = color_srgb_to_linear_backward_array(label)

        mask  = np.array(Image.open(self.list_mask[index]))
        mask_dict = {3: [255, 0, 0], 0:[0, 0, 255], 1:[255, 255, 0], 2:[255,0, 255]}
        meta = scipy.io.loadmat(self.list_meta[index])

        choose_to_whole = []
        for i in range(num_parts):
            choose_to_whole.append( np.where( (mask[:, :, 0] == mask_dict[i][0]) & (mask[:, :, 1] == mask_dict[i][1]) & (mask[:, :, 2] == mask_dict[i][2])))

        # backproject pcloud
        C = meta['C']
        P = meta['projection_matrix']
        RT = meta['rotation_translation_matrix']
        K = meta['intrinsic_matrix']
        cloud_in_cam, _ = self.backproject(rgb2gray(depth), C, P, RT, K)

        for s in range(num_parts):
            x_set, y_set   = choose_to_whole[s]
            if len(x_set)<2:
                print('data is empty, skipping!!!')
                return None

            parts_cloud_cam[s]    = cloud_in_cam[x_set, y_set, :3]
            parts_cloud_world[s]  = (np.dot(RT[:3, :3].T, parts_cloud_cam[s].T - RT[:, 3:4])).T
            parts_cloud_urdf[s]   = label[x_set, y_set, :3]

        # transform joints/contacts into camera space
        joints_cam   =  (np.dot(RT[:3, :3], joints.T)   + RT[:, 3:4]).T
        contacts_cam =  (np.dot(RT[:3, :3], contacts.T) + RT[:, 3:4]).T

        # save into h5 for rgb_img, input_pts, mask, correpsonding urdf_points
        print('Writing to ', h5_save_name)
        hf = h5py.File(h5_save_name, 'w')
        hf.create_dataset('joints', data=joints_cam)
        hf.create_dataset('contacts', data=contacts_cam)
        cloud_cam=hf.create_group('gt_points')
        for part_i, points in enumerate(parts_cloud_cam):
            cloud_cam.create_dataset(str(part_i), data=points)
        coord_gt=hf.create_group('gt_coords')
        for part_i, points in enumerate(parts_cloud_urdf):
            coord_gt.create_dataset(str(part_i), data=points)
        hf.close()

        ################# for debug only, let me know if you have questions #################
        # if self.is_debug:
        #     figure = plt.figure(dpi=200)
        #     ax = plt.subplot(131)
        #     plt.imshow(img)
        #     plt.title('RGB image')
        #     ax1 = plt.subplot(132)
        #     plt.imshow((depth/2/np.max(depth))[:, :, :])
        #     plt.title('depth image')
        #     ax2 = plt.subplot(133)
        #     plt.imshow(label)
        #     plt.title('NOCS image')
        #     plt.show()
        #     # plot3d_pts([parts_cloud_cam+[joints_cam], [joints_cam]], [['part {}'.format(i) for i in range(len(parts_cloud_cam)+1)], ['joints']], s=5, title_name=['camera coords', 'joints'], save_fig=True)
        #     plot3d_pts([parts_cloud_cam+[contacts_cam], [contacts_cam]], [['part {}'.format(i) for i in range(len(parts_cloud_cam)+1)], ['contacts']], s=5, title_name=['camera coords', 'joints'], save_fig=True)
        #     plot3d_pts([parts_cloud_world+[joints], [joints]], [['part {}'.format(i) for i in range(len(parts_cloud_world)+1)], ['joints']], s=5, title_name=['world coords', 'joints'], save_fig=False)
        #     # plot3d_pts([parts_cloud_canon], [['part {}'.format(i) for i in range(len(parts_cloud_canon))]], s=5, title_name=['canon coords'], save_fig=True)
        #     # plot3d_pts([parts_cloud_urdf], [['part {}'.format(i) for i in range(len(parts_cloud_urdf))]], s=5, title_name=['NOCS coords'], save_fig=True)

        return None

    def preprocess_batch(self, s_ind, e_ind):
        # 2. preprocess and save
        for i in range(s_ind, e_ind):
            data = self.__preprocess_and_save__(i)


if __name__ == '__main__':
    #>>>>>>>>>>>>>>>>>>>>>>>>> config here >>>>>>>>>>>>>>>>>>>>>>>#
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='shape2motion', help='name of the dataset we use') # todo
    parser.add_argument('--item', default='eyeglasses', help='name of the dataset we use')
    parser.add_argument('--num_expr', default='0.01', help='get configuration file per expriment')
    parser.add_argument('--mode', default='train', help='indicating whether in demo mode')
    parser.add_argument('--debug', action='store_true', help='indicating whether in debug mode')
    args = parser.parse_args()

    name_dataset  = args.dataset
    args.debug    = False
    item          = args.item
    my_dir        = infos.base_path

    root_dset     = my_dir + '/dataset/' + name_dataset
    render_path   = infos.render_path
    second_dset   = infos.second_path + '/data'

    selected_list = infos.datasets[item].train_list # default None, if specifies, will only choose specified instances
    #>>>>>>>>>>>>>>>>>>>>>>>>> config end here >>>>>>>>>>>>>>>>>>>#

    PoseData      = PoseDataset(root_dset, render_path, item,  second_dir=second_dset, is_debug=args.debug, mode=args.mode, selected_list=selected_list)
    print('number of images: ', len(PoseData.list_rgb))

    # multi-processing
    starttime   = time.time()
    processes   = []
    cpuCount    = os.cpu_count() - 1
    if args.debug:
        num_per_cpu = 10
    else:
        num_per_cpu = int(len(PoseData)/cpuCount) + 1

    for k in range(cpuCount):
        e_ind = min(num_per_cpu*(k+1), len(PoseData))
        p=Process(target=PoseData.preprocess_batch, args=(num_per_cpu*k, e_ind))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('Process {} took {} seconds, with average {} seconds per data'.format(num_per_cpu*cpuCount, time.time() - starttime, (time.time() - starttime)/(num_per_cpu*cpuCount) ))

    # # 3. split data into train & test
    # split_dataset(second_dset, [item], args, test_ins=infos.datasets[item].test_list, spec_ins=infos.datasets[item].spec_list, train_ins=infos.datasets[item].train_list)
