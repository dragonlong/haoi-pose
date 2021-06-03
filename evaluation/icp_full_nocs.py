import os
import time
from os import makedirs, remove
from os.path import exists, join
from time import time
import torch

import open3d as o3d
import numpy as np
from os.path import join as pjoin
import copy
from glob import glob
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt  # matplotlib.use('Agg') # TkAgg
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

import __init__
from global_info import global_info

infos     = global_info()
my_dir    = infos.base_path
second_path = infos.second_path
render_path = infos.render_path
viz_path  = infos.viz_path
hand_mesh = infos.hand_mesh
hand_urdf = infos.hand_urdf
grasps_meta  = infos.grasps_meta
mano_path    = infos.mano_path

whole_obj = infos.whole_obj
part_obj  = infos.part_obj
obj_urdf  = infos.obj_urdf
categories_id = infos.categories_id
project_path = infos.project_path
def bp():
    import pdb;pdb.set_trace()

def draw_registration_result(source, target, transformation, use_pv=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    if not use_pv:
        o3d.visualization.draw_geometries([source_temp, target_temp])
    else:
        p = pv.Plotter(off_screen=off_screen, lighting='light_kit')
        for sample in [np.asarray(source_temp.points), np.asarray(target_temp.points),]:
            point_cloud = pv.PolyData(sample)
            p.add_mesh(point_cloud, color=np.random.rand(3), point_size=15, render_points_as_spheres=True)

        p.add_title('transformed source + target', font_size=16)
        p.show_grid()
        p.show()

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_in, target_in, voxel_size, load=False, viz=False):
    if load:
        print(":: Load two point clouds and disturb initial pose.")
        source = o3d.io.read_point_cloud(source_in, format='xyz')
        target = o3d.io.read_point_cloud(target_in, format='xyz')
    else:
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_in)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(source_in)
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    if viz:
        draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    # result = o3d.pipelines.registration.registration_icp(
    #     source, target, distance_threshold, result_ransac.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    result =  o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return result

def get_txt(fn):
    point_normal_set = np.loadtxt(fn, delimiter=' ').astype(np.float32)
    return point_normal_set

def plot_arrows(fn=None, r_mat=None):
    ###############################################################################
    # fn = '/home/dragon/Documents/ICML2021/results/preds/train_0_0002_0_0.txt'
    # fn = '/home/dragon/Documents/ICML2021/results/preds/0.59/train_2850_0002_850_0.txt'
    # fn = '/home/dragon/Documents/ICML2021/results/preds/0.63/train_990_0002_0_0.txt'
    if fn is None:
        fn = '/home/dragon/Documents/ICML2021/results/preds/0.61/train_1000_0002_0_0.txt'
    point_normal_set = np.loadtxt(fn, delimiter=' ').astype(np.float32)
    points = point_normal_set[:, :3]
    r_mat  = point_normal_set[:, 4:].reshape(-1, 3, 3)
    x_axis = np.matmul(np.array([[1.0, 0.0, 0.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
    y_axis = np.matmul(np.array([[0.0, 1.0, 0.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
    z_axis = np.matmul(np.array([[0.0, 0.0, 1.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
    point_cloud = pv.PolyData(points)

    # cloud['point_color'] = cloud.points[:, 2]  # just use z coordinate
    # pv.plot(cloud, scalars='point_color', cmap='jet', show_bounds=True, cpos='yz')
    ###############################################################################
    point_cloud['vectors'] = x_axis[:, 0, :]
    ###############################################################################.
    # cent = np.random.random((100, 3))
    # direction = np.random.random((100, 3))
    # pyvista.plot_arrows(cent, direction)
    arrows = point_cloud.glyph(orient='vectors', scale=False, factor=0.15,)

    # Display the arrows
    p = pv.Plotter()
    p.add_mesh(point_cloud, color='maroon', point_size=10.,
                     render_points_as_spheres=True)
    p.add_mesh(arrows, color='blue')
    point_cloud['vectors'] = y_axis[:, 0, :]
    arrows1 = point_cloud.glyph(orient='vectors', scale=False, factor=0.15,)
    p.add_mesh(arrows1, color='red')
    p.add_point_labels([point_cloud.center,], ['Center',],
                             point_color='yellow', point_size=20)
    # sphere = pv.Sphere(radius=3.14)
    # sphere.vectors = vectors * 0.3
    p.show_grid()
    p.show()


def get_tableau_palette():
    palette = np.array([[ 78,121,167], # blue
                        [255, 87, 89], # red
                        [ 89,169, 79], # green
                        [242,142, 43], # orange
                        [237,201, 72], # yellow
                        [176,122,161], # purple
                        [255,157,167], # pink
                        [118,183,178], # cyan
                        [156,117, 95], # brown
                        [186,176,172]  # gray
                        ],dtype=np.uint8)
    return palette

def set_axes_equal(ax, limits=None):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    if limits is None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    else:
        x_limits, y_limits, z_limits = limits
        ax.set_xlim3d([x_limits[0], x_limits[1]])
        ax.set_ylim3d([y_limits[0], y_limits[1]])
        ax.set_zlim3d([z_limits[0], z_limits[1]])

def plot_distribution(d, labelx='Value', labely='Frequency', title_name='Mine', dpi=200, xlimit=None, put_text=False, save_fig=False, sub_name='seen'):
    fig     = plt.figure(dpi=dpi)
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title_name)
    if put_text:
        plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if xlimit is not None:
        plt.xlim(xmin=xlimit[0], xmax=xlimit[1])
    plt.show()
    if save_fig:
        if not os.path.exists('./results/test/'):
            os.makedirs('./results/test/')
        print('--saving fig to ', './results/test/{}_{}.png'.format(title_name, sub_name))
        fig.savefig('./results/test/{}_{}.png'.format(title_name, sub_name), pad_inches=0)
    plt.close()

def plot3d_pts(pts, pts_name, s=1, dpi=350, title_name=None, sub_name='default', arrows=None, \
                    color_channel=None, colorbar=False, limits=None,\
                    bcm=None, puttext=None, view_angle=None,\
                    save_fig=False, save_path=None, save_name=None, flip=True,\
                    axis_off=False, show_fig=True, mode='pending'):
    """
    fig using,
    """
    fig     = plt.figure(dpi=dpi)
    # cmap    = plt.cm.jet
    top     = plt.cm.get_cmap('Oranges_r', 128)
    bottom  = plt.cm.get_cmap('Blues', 128)
    if isinstance(s, list):
        ss = s
    else:
        ss = [s] * len(pts[0])
    colors = np.vstack((top(np.linspace(0, 1, 10)),
                           bottom(np.linspace(0, 1, 10))))
    # colors = ListedColormap(newcolors, name='OrangeBlue')
    # colors  = cmap(np.linspace(0., 1., 5))
    # '.', '.', '.',
    all_poss=['o', 'o', 'o', 'o','o', '*', '.','o', 'v','^','>','<','s','p','*','h','H','D','d','1','','']
    c_set   = ['r', 'b', 'g', 'k', 'm']
    arrow_len = [0.25, 0.45]
    num     = len(pts)
    for m in range(num):
        ax = plt.subplot(1, num, m+1, projection='3d')
        if view_angle==None:
            ax.view_init(elev=11, azim=-132)
        else:
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
        # if len(pts[m]) > 1:
        for n in range(len(pts[m])):
            if color_channel is None:
                ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2], marker=all_poss[n], s=ss[n], cmap=colors[n], label=pts_name[m][n], depthshade=False)
            else:
                if len(color_channel[m][n].shape) < 2:
                    color_channel[m][n] = color_channel[m][n][:, np.newaxis] * np.array([[1]])
                if np.amax(color_channel[m][n], axis=0, keepdims=True)[0, 0] == np.amin(color_channel[m][n], axis=0, keepdims=True)[0, 0]:
                    rgb_encoded = color_channel[m][n]
                else:
                    rgb_encoded = (color_channel[m][n] - np.amin(color_channel[m][n], axis=0, keepdims=True))/np.array(np.amax(color_channel[m][n], axis=0, keepdims=True) - np.amin(color_channel[m][n], axis=0, keepdims=True)+ 1e-6)
                if len(pts[m])==3 and n==2:
                    p = ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2],  marker=all_poss[4], s=ss[n], c=rgb_encoded, label=pts_name[m][n], depthshade=False)
                else:
                    p = ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2],  marker=all_poss[n], s=ss[n], c=rgb_encoded, label=pts_name[m][n], depthshade=False)
                if colorbar:
                    fig.colorbar(p)
            if arrows is not None:
                points, offset_sub = arrows[m][n]['p'], arrows[m][n]['v']
                offset_sub = offset_sub * arrow_len[n]
                if len(points.shape) < 2:
                    points = points.reshape(-1, 3)
                if len(offset_sub.shape) < 2:
                    offset_sub = offset_sub.reshape(-1, 3)
                if offset_sub.shape[0] == 3:
                    ax.quiver(points[0:1, 0], points[0:1, 1], points[0:1, 2], offset_sub[0:1, 0], offset_sub[0:1, 1], offset_sub[0:1, 2], color='r', linewidth=2)
                    ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset_sub[1:2, 0], offset_sub[1:2, 1], offset_sub[1:2, 2], color='g', linewidth=2)
                    ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset_sub[2:3, 0], offset_sub[2:3, 1], offset_sub[2:3, 2], color='b', linewidth=2)
                else:
                    ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset_sub[:, 0], offset_sub[:, 1], offset_sub[:, 2], color=c_set[n], linewidth=4)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        if axis_off:
            plt.axis('off')

        if title_name is not None:
            if len(pts_name[m])==1:
                plt.title(title_name[m]+ ' ' + pts_name[m][0] + '    ')
            else:
                plt.legend(loc=0)
                plt.title(title_name[m]+ '    ')

        if bcm is not None:
            for j in range(len(bcm)):
                ax.plot3D([bcm[j][0][0], bcm[j][2][0], bcm[j][6][0], bcm[j][4][0], bcm[j][0][0]], \
                    [bcm[j][0][1], bcm[j][2][1], bcm[j][6][1], bcm[j][4][1], bcm[j][0][1]], \
                    [bcm[j][0][2], bcm[j][2][2], bcm[j][6][2], bcm[j][4][2], bcm[j][0][2]], 'blue')

                ax.plot3D([bcm[j][1][0], bcm[j][3][0], bcm[j][7][0], bcm[j][5][0], bcm[j][1][0]], \
                    [bcm[j][1][1], bcm[j][3][1], bcm[j][7][1], bcm[j][5][1], bcm[j][1][1]], \
                    [bcm[j][1][2], bcm[j][3][2], bcm[j][7][2], bcm[j][5][2], bcm[j][1][2]], 'gray')

                for pair in [[0, 1], [2, 3], [4, 5], [6, 7]]:
                    ax.plot3D([bcm[j][pair[0]][0], bcm[j][pair[1]][0]], \
                        [bcm[j][pair[0]][1], bcm[j][pair[1]][1]], \
                        [bcm[j][pair[0]][2], bcm[j][pair[1]][2]], 'red')
        if puttext is not None:
            ax.text2D(0.55, 0.80, puttext, transform=ax.transAxes, color='blue', fontsize=6)
        # if limits is None:
        #     limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
        set_axes_equal(ax, limits=limits)
        # cam_equal_aspect_3d(ax, np.concatenate(pts[0], axis=0), flip_x=flip, flip_y=flip)
    if show_fig:
        if mode == 'continuous':
            plt.draw()
        else:
            plt.show()

    if save_fig:
        if (save_path is None) and (save_name is None):
            if not os.path.exists('./results/test/'):
                os.makedirs('./results/test/')
            fig.savefig('./results/test/{}_{}.png'.format(sub_name, title_name[0]), pad_inches=0)
        else:
            if save_name is not None:
                fig.savefig(save_name, pad_inches=0)
            else:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fig.savefig('{}/{}_{}.png'.format(save_path, sub_name, title_name[0]), pad_inches=0)
    if mode != 'continuous':
        plt.close()


# def global_registration(source_name, target_name, voxel_size, use_anchors=True):
#     """
#         # for fn in fnames['canon']:
#         #     source_name = fn
#         #     target_name = fn.replace('canon', 'input')
#
#     source_name: txt file for predicted canonical shape;
#     target_name: txt file for input point cloud;
#     """
#     result_ransac = execute_global_registration(source_down, target_down,
#                                                 source_fpfh, target_fpfh,
#                                                 voxel_size)
#     print(result_ransac.transformation)
#     draw_registration_result(source_down, target_down,
#                              result_ransac.transformation)
#
# def local_icp(source, target, source_fpfh, target_fpfh, voxel_size):
#     result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
#                                      voxel_size)
#     print('2. ', np.asarray(target.points).mean())
#     print(result_icp.transformation)
#     draw_registration_result(source, target, result_icp.transformation)
#     new_pcd = copy.deepcopy(source).transform(result_icp.transformation)
#     transformed_pts = np.asarray(new_pcd.points)
#     # np.savetxt(fn.replace('canon', 'icp'), transformed_pts)
#     # print('saving to', fn.replace('canon', 'icp'))

def load_input(cfg):
    res_path = f'{my_dir}/results/test_pred/{cfg.name_dset}/{cfg.exp_num}_unseen_part_rt_pn_general.npy'
    results = np.load(res_path, allow_pickle=True).item()
    infos_dict, track_dict = results['info'], results['err']
    return infos_dict, track_dict

def rot_diff_rad(rot1, rot2, chosen_axis=None):
    if chosen_axis is not None:
        axis = {'x': 0, 'y': 1, 'z': 2}[chosen_axis]
        y1, y2 = rot1[..., axis], rot2[..., axis]  # [Bs, 3]
        diff = np.sum(y1 * y2, axis=-1)  # [Bs]
    else:
        mat_diff = np.matmul(rot1, rot2.swapaxes(-1, -2))
        diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
        diff = (diff - 1) / 2.0
    diff = np.clip(diff, a_min=-1.0, a_max=1.0)
    return np.arccos(diff)


def rot_diff_degree(rot1, rot2, chosen_axis=None):
    return rot_diff_rad(rot1, rot2, chosen_axis=chosen_axis) / np.pi * 180.0

def draw_geo(geo_list):
    """
    pcd_list = []
    for geo in geo_list:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(geo)
        pcd_list.append(pcd)

    o3d.visualization.draw_geometries(pcd_list)
    """
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    for geo in geo_list:
        ax.scatter(geo[..., 0], geo[..., 1], geo[..., 2])
    plt.show()


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def base_icp(source, target, init_trans, method='point2plane', threshold=0.05, max_iter=2000):
    """threshold: assume inputs are normalized; init may have a large translation error"""
    method_dict = {'point2plane': o3d.pipelines.registration.TransformationEstimationPointToPlane,
                   'point2point': o3d.pipelines.registration.TransformationEstimationPointToPoint}
    method = method_dict[method]()
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_trans, method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))


def eval_registeration(target, source, trans):
    """
    set threshold to 1.0 -> want to compute the rmse for all target points
    note that evaluate_registration will find correspondences for the *first* argument
    """
    result = o3d.pipelines.registration.evaluate_registration(target, source,
                                                              1.0, trans)
    return result.inlier_rmse


def get_init_pose(source, target, method='rotate_60', voxel_size=0.05):
    if method == 'global':
        source_down, source_fpfh = preprocess_point_cloud(target, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(source, voxel_size)
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        init_poses = np.expand_dims(np.asarray(result_ransac.transformation), 0)
    elif method == 'rotate_60':
        def get_bbox_center(pts):  # pts: [...., 3, N]
            return (np.min(pts, axis=-1, keepdims=True) +
                    np.max(pts, axis=-1, keepdims=True)) * 0.5

        target = np.asarray(target.points)
        target = target.reshape((1, ) + target.shape).transpose(0, 2, 1).repeat(
            len(anchors), 0)
        rotated_target = torch.matmul(torch.tensor(anchors).float(),
                                       torch.tensor(target).float()).numpy()
        centers = get_bbox_center(rotated_target)
        init_poses = np.zeros((len(anchors), 4, 4))
        init_poses[..., :3, :3] = anchors
        init_poses[..., :3, 3:] = -centers
        init_poses[..., 3, 3] = 1
    else:
        assert 0, f'Unsupported method {method}'

    return init_poses


def icp_registeration(source, target, init_method='rotate_60', voxel_size=0.05,
                      icp_method='point2point', icp_threshold=0.05, icp_max_iter=2000):
    init_poses = get_init_pose(copy.deepcopy(source), copy.deepcopy(target),
                               method=init_method, voxel_size=voxel_size)
    best_pose, min_rmse = None, 1e9
    for init_pose in init_poses:
        cur_result = base_icp(copy.deepcopy(target), copy.deepcopy(source), init_pose,
                              method=icp_method, threshold=icp_threshold, max_iter=icp_max_iter)
        cur_rmse = eval_registeration(target, source, cur_result.transformation)
        if cur_rmse < min_rmse:
            min_rmse = cur_rmse
            best_pose = cur_result.transformation
    return best_pose


def eval_pose(pose, r, t, chosen_axis=None):
    rdiff = rot_diff_degree(pose[:3, :3], r, chosen_axis=chosen_axis)
    tdiff = pose[:3, 3] - t
    tdiff = np.sqrt((tdiff ** 2).sum())

    return {'rdiff': rdiff, 'tdiff': tdiff,
            '5deg': float(rdiff <= 5), '5cm': float(tdiff <= 0.05),
            '5deg5cm': float(rdiff <= 5 and tdiff <= 0.05)}

class simple_config(object):
    def __init__(self, target_category='airplane', name_dset='modelnet40aligned', icp_method_type=0):
        self.log_dir = 'default'
        self.icp_method_type  = icp_method_type    # -1 for predicted shape, 0 for example shape, 1 for GT shape
        self.symmetry_type    = 0    # 0 for non-symmetric, 1 for symmetric;
        self.name_dset = name_dset
        self.target_category=target_category
        self.chosen_axis= None

        if name_dset == 'modelnet40aligned':
            self.dataset_path=f'{my_dir}/data/modelnet40aligned/EvenAlignedModelNet40PC'
            if self.target_category == 'airplane':
                self.exp_num    = '0.813'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'car':
                self.exp_num    = '0.851'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'chair':
                self.exp_num    = '0.8581'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'sofa':
                self.exp_num    = '0.859'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'bottle':
                self.exp_num    = '0.8562'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                self.symmetry_type = 1
                self.chosen_axis = 'z'
        elif name_dset == 'modelnet40new':
            self.dataset_path=f'{my_dir}/data/modelnet40new/render/{target_category}/test/gt'
            if self.target_category == 'airplane':
                self.exp_num    = '0.913r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'car':
                self.exp_num    = '0.921r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'chair':
                self.exp_num    = '0.951r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'sofa':
                self.exp_num    = '0.96r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'bottle':
                self.exp_num    = '0.941r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                self.symmetry_type = 1
                self.chosen_axis = 'z'
        elif name_dset == 'nocs_newer':
            self.dataset_path=f'{my_dir}/data/modelnet40new/render/{target_category}/test/gt'
            if self.target_category == 'airplane':
                self.exp_num    = '0.913r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'car':
                self.exp_num    = '0.921r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'chair':
                self.exp_num    = '0.951r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'sofa':
                self.exp_num    = '0.96r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
            elif self.target_category == 'bottle':
                self.exp_num    = '0.941r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                self.symmetry_type = 1
                self.chosen_axis = 'z'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_dset', default='modelnet40aligned', help='yyds')
    parser.add_argument('--target_category', default='sofa', help='count of articulation change')
    parser.add_argument('--icp_method_type', default=0, type=int, help='icp_method_type')
    args = parser.parse_args()

    voxel_size = 0.02  # means 5cm for the dataset
    pv.set_plot_theme("document")
    off_screen = False
    color      = 'gold' #'deepskyblue'
    window_shape = (1,1)
    k = 1.3
    k1 = 1.05
    font_size    = 18
    query_keys = ['canon', 'target', 'input', 'pred']
    # cfg = simple_config()
    # cfg = simple_config(exp_num='0.851', target_category='car')
    # cfg = simple_config(exp_num='0.85', target_category='chair')
    cfg = simple_config( target_category=args.target_category, name_dset=args.name_dset, icp_method_type=args.icp_method_type)

    # fpath   = f'{my_dir}results/preds/{cfg.exp_num}/generation/' # generation/generation
    # if exists(fpath):
    #     fnames  = {}
    #     for key in query_keys:
    #         fnames[key] = sorted(glob(f'{fpath}/*{key}*txt'))

    # load npy '/model/${item}/${exp_num}'
    infos_dict, track_dict = load_input(cfg)
    basenames = infos_dict['basename']
    inputs   = np.array(infos_dict['in'])
    r_gt     = np.array(infos_dict['r_gt'])
    t_gt     = np.array(infos_dict['t_gt'])
    r_pred   = np.array(infos_dict['r_pred'])
    rdiff    = np.array(track_dict['rdiff'])
    num      = len(basenames)
    print(f'---we have {num} samples')

    # load canonical shape
    all_data = []
        pc = np.random.permutation(instance_points)[:1024]

    boundary_pts = [np.min(pc, axis=0), np.max(pc, axis=0)]
    center_pt = (boundary_pts[0] + boundary_pts[1])/2
    length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
    pc_canon = (pc - center_pt.reshape(1, 3))/length_bb

    err_dict = {'global': {}, 'rotate_60': {}}

    anchors = np.load('anchors.npy') # 60, 3, 3 as initialization
    for i in range(inputs.shape[0]):
        target_pts, source_pts = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        target_pts.points = o3d.utility.Vector3dVector(inputs[i])
        source_pts.points = o3d.utility.Vector3dVector(pc_canon)
        for init_method in ['rotate_60']:
            pose = icp_registeration(copy.deepcopy(source_pts),
                                     copy.deepcopy(target_pts),
                                     init_method=init_method)
            pred_pose = np.linalg.inv(pose)
            # draw_registration_result(source_pts, target_pts, pred_pose, use_pv=True)
            err = eval_pose(pred_pose, r_gt[i], t_gt[i], chosen_axis=cfg.chosen_axis)
            print(f'--{i}th ', err)
            cur_dict = err_dict[init_method]
            for key, value in err.items():
                if key not in cur_dict:
                    cur_dict[key] = []
                cur_dict[key].append(float(value))

    final_dict = {}
    mid_dict   = {}
    for init_method in ['rotate_60']:
        cur_dict = err_dict[init_method]
        final_dict[init_method] = {}
        cur_final = final_dict[init_method]
        for key, value in cur_dict.items():
            cur_final[key] = np.mean(np.array(value))
            if key == 'rdiff':
                cur_final['rdiff_mid'] = np.median(np.array(value))
    print(f'>>>>>>>>>>>>>>>>--{cfg.name_dset}--{cfg.target_category}--<<<<<<<<<<<<<<<<<<')
    for init_method in ['rotate_60']:
        print('\n---Init:', init_method)
        for key, value in final_dict[init_method].items():
            print(f'{key}: {value}')

    # print('Con!!!')
