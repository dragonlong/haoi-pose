import os
import pickle

import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from igl import read_triangle_mesh, signed_distance
from meshplot import plot, subplot, interact

import __init__
from global_info import global_info
from common.queries import BaseQueries, get_trans_queries
from common import handutils
from common.data_utils import fast_load_obj
from common.d3_utils import calculate_3d_backprojections
from common.vis_utils import plot_imgs
from dataset.obman import ObMan
ImageFile.LOAD_TRUNCATED_IMAGES = True
def breakpoint():
    import pdb;pdb.set_trace()

# try using custom packages
infos     = global_info()
my_dir    = infos.base_path
group_path= infos.group_path
second_path = infos.second_path
render_path = infos.render_path
project_path= infos.project_path

class ObMan_HO(ObMan):
    def __init__(
        self,
        split="train",
        mode="obj",
        joint_nb=21,
        mini_factor=None,
        use_cache=False,
        root_palm=False,
        segment=False,
        override_scale=False,
        use_external_points=True,
        apply_obj_transform=True,
        segmented_depth=True,
        shapenet_root="datasymlinks/ShapeNetCore.v2",
        obman_root="datasymlinks/obman"):
        ObMan.__init__(self,
                        split=split,
                        mode=mode,
                        joint_nb=joint_nb,
                        mini_factor=mini_factor,
                        use_cache=use_cache,
                        root_palm=root_palm,
                        segment=segment,
                        override_scale=override_scale,
                        use_external_points=use_external_points,
                        apply_obj_transform=apply_obj_transform,
                        segmented_depth=segmented_depth,
                        shapenet_root=shapenet_root,
                        obman_root=obman_root)
    def get_surface_pts(self, idx, canon_pts=None, debug=False):
        # read obj
        model_path = self.obj_paths[idx]
        surface_pts_path = model_path.replace("model_normalized", "surface_points")
        all_pts = np.load(surface_pts_path, allow_pickle=True)

        # sample_path = self.image_names[idx]
        # hand_surface_path = sample_path.replace('rgb', 'occupancy').replace('.jpg', '_hand_surface.pkl')
        #
        # change target points
        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # we care about the nomralized shape in NOCS
        points = (all_pts - center_pt.reshape(1, 3)) / length_bb + 0.5
        nomral_vectors = np.ones_like(points)
        return points, nomral_vectors

    def get_occupany_pts(self, idx, canon_pts=None, npt=2048, thres=0.01, debug=False):
        # read obj
        model_path = self.obj_paths[idx]
        outside_pts_path = model_path.replace("model_normalized", "outside_points")
        inside_pts_path  = model_path.replace("model_normalized", "inside_points")
        surface_pts_path = model_path.replace("model_normalized", "surface_points")

        sample_path = self.image_names[idx]
        hand_surface_path = sample_path.replace('rgb', 'occupancy').replace('.jpg', '_hand_surface.pkl')
        hand_outside_path = sample_path.replace('rgb', 'occupancy').replace('.jpg', '_hand_outside.pkl')
        hand_inside_path  = sample_path.replace('rgb', 'occupancy').replace('.jpg', '_hand_inside.pkl')
        # if os.path.exists(model_path):
        #     with open(model_path, "rb") as obj_f:
        #         mesh = pickle.load(obj_f)
        all_pts = []
        filenames = [inside_pts_path, surface_pts_path, hand_inside_path, hand_surface_path, hand_outside_path]
        for filename in filenames:
            if os.path.exists(filename):
                all_pts.append(np.load(filename, allow_pickle=True))
        if len(all_pts) < 5:
            print('hand pts error, check ', idx)
            return np.ones([2048, 3]) * 0.5, np.array([[1, 0]]) * np.ones((2048, 2))

        length_list = [256, 256, 256, 256, 1024]
        o_value     = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0]])
        final_pts   = []
        final_occu  = []
        contacts_path = sample_path.replace('rgb', 'occupancy').replace('.jpg', '_contacts.npz')
        if os.path.exists(contacts_path):
            contact_dict = np.load(contacts_path)
            if 'null' not in contact_dict:
                hand_contacts_index = contact_dict['hand']
                object_contacts_index = contact_dict['object']
            else:
                hand_contacts_index = []
                object_contacts_index = []
        else:
            hand_contacts_index = []
            object_contacts_index = []
        for j in range(5):
            inds = np.random.choice(all_pts[j].shape[0], min(length_list[j], all_pts[j].shape[0]))
            occupancy_label = np.zeros((all_pts[j].shape[0], 2))
            occupancy_label[:, 0] = o_value[j, 0]
            occupancy_label[:, 1] = o_value[j, 1]
            if j == 1 and len(object_contacts_index) > 0:
                occupancy_label[object_contacts_index, 1] = 1
            if j == 3 and len(hand_contacts_index) > 0:
                occupancy_label[hand_contacts_index, 0] = 1
            final_pts.append(all_pts[j][inds])
            final_occu.append(occupancy_label[inds])
            if len(inds) < length_list[j]:
                length_list[j+1] += length_list[j] - len(inds)

        # occupancy_labels = np.zeros((2048, 2))
        # num1 = length_list[0] + length_list[1]
        # occupancy_labels[:num1, 0] = 1
        # num2 = sum(length_list[:4])
        # occupancy_labels[num1:num2, 1] = 1

        points = np.concatenate(final_pts, axis=0)
        occupancy_labels = np.concatenate(final_occu, axis=0)

        # change target points
        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # we care about the nomralized shape in NOCS
        points = (points - center_pt.reshape(1, 3)) / length_bb + 0.5

        return points, occupancy_labels

    def get_volume_pts(self, idx, canon_pts=None, npt=2048, thres=0.01):
        model_path = self.obj_paths[idx]
        outside_pts_path = model_path.replace("model_normalized", "outside_points")
        inside_pts_path  = model_path.replace("model_normalized", "inside_points")
        # if os.path.exists(model_path):
        #     with open(model_path, "rb") as obj_f:
        #         mesh = pickle.load(obj_f)

        all_pts = []
        filenames = [inside_pts_path, outside_pts_path]
        for filename in filenames:
            if os.path.exists(filename):
                # with open(filename, "r") as m_f:
                #     all_pts.append(pickle.load(m_f))
                all_pts.append(np.load(filename, allow_pickle=True))

        occupancy_labels0 = np.ones((all_pts[0].shape[0]))
        occupancy_labels1 = np.zeros((all_pts[1].shape[0]))
        occupancy_labels = np.concatenate([occupancy_labels0, occupancy_labels1], axis=0)
        points = np.concatenate(all_pts, axis=0)

        # change target points
        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # we care about the nomralized shape in NOCS
        points = (points - center_pt.reshape(1, 3)) / length_bb + 0.5

        return points, occupancy_labels
