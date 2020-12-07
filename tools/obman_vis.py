import os
import yaml
import numpy as np
import time
import h5py
import pickle

from random import randint, sample
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import __init__
from common.vis_utils import cam_equal_aspect_3d
from common import data_utils, handutils, vis_utils
from common.queries import (
    BaseQueries,
    TransQueries,
    one_query_in,
    no_query_in,
)

def visualize_img(sample, joint_idxs=False):
    links = [
                (0, 1, 2, 3, 4),
                (0, 5, 6, 7, 8),
                (0, 9, 10, 11, 12),
                (0, 13, 14, 15, 16),
                (0, 17, 18, 19, 20),
            ]
    img = sample[BaseQueries.images]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Add title
    if BaseQueries.sides in sample:
        side = sample[BaseQueries.sides]
        ax.set_title("{} hand".format(side))
    ax.imshow(img)
    if BaseQueries.joints2d in sample:
        joints = sample[BaseQueries.joints2d]
        # Scatter hand joints on image
        vis_utils.visualize_joints_2d(
            ax, joints, joint_idxs=False, links=links
        )
        ax.axis("off")
    if BaseQueries.objpoints2d in sample:
        objpoints = sample[BaseQueries.objpoints2d]
        # Scatter hand joints on image
        ax.scatter(objpoints[:, 0], objpoints[:, 1], alpha=0.01)
    plt.show()

def display_3d(ax, sample, proj="z", joint_idxs=False, axis_off=False):
    # Scatter  projection of 3d vertices
    pts = []
    if TransQueries.verts3d in sample:
        verts3d = sample[TransQueries.verts3d]
        pts.append(verts3d)
        ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], s=1)

    # Scatter projection of object 3d vertices
    if TransQueries.objpoints3d in sample:
        obj_verts3d = sample[TransQueries.objpoints3d]
        pts.append(obj_verts3d)
        ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

    # Scatter  projection of 3d vertices
    if BaseQueries.verts3d in sample:
        verts3d = sample[BaseQueries.verts3d]
        pts.append(verts3d)
        ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], s=1)

    # Scatter projection of object 3d vertices
    if BaseQueries.objpoints3d in sample:
        obj_verts3d = sample[BaseQueries.objpoints3d]
        pts.append(obj_verts3d)
        ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

    # Scatter projection of object 3d vertices
    if BaseQueries.pcloud in sample:
        obj_verts3d = sample[BaseQueries.pcloud]
        pts.append(obj_verts3d)
        ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

    # Scatter projection of object 3d vertices
    if TransQueries.pcloud in sample:
        obj_verts3d = sample[TransQueries.pcloud]
        pts.append(obj_verts3d)
        ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

    cam_equal_aspect_3d(ax, np.concatenate(pts, axis=0))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if axis_off:
        plt.axis('off')

def display_pts(ax, verts3d, tn='pts', proj="z", joint_idxs=False, axis_off=False):
    # Scatter  projection of 3d vertices
    pts = []
    pts.append(verts3d)
    ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], s=1)

    cam_equal_aspect_3d(ax, np.concatenate(pts, axis=0))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if tn is not None:
        plt.title(tn)
    if axis_off:
        plt.axis('off')

def display_proj(ax, sample, proj="z", joint_idxs=False):
    if proj == "z":
        proj_1 = 0
        proj_2 = 1
        ax.invert_yaxis()
    elif proj == "y":
        proj_1 = 0
        proj_2 = 2
    elif proj == "x":
        proj_1 = 1
        proj_2 = 2

    if TransQueries.joints3d in sample:
        joints3d = sample[TransQueries.joints3d]
        vis_utils.visualize_joints_2d(
            ax,
            np.stack([joints3d[:, proj_1], joints3d[:, proj_2]], axis=1),
            joint_idxs=joint_idxs,
            links=self.pose_dataset.links,
        )
    # Scatter  projection of 3d vertices
    if TransQueries.verts3d in sample:
        verts3d = sample[TransQueries.verts3d]
        ax.scatter(verts3d[:, proj_1], verts3d[:, proj_2], s=1)

    # Scatter projection of object 3d vertices
    if TransQueries.objpoints3d in sample:
        obj_verts3d = sample[TransQueries.objpoints3d]
        ax.scatter(obj_verts3d[:, proj_1], obj_verts3d[:, proj_2], s=1)
    ax.set_aspect("equal")  # Same axis orientation as imshow

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
