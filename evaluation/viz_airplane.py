# import numpy as np
import numpy as np
import os
import sys
import random
import hydra
import h5py
import torch
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf
from os import makedirs, remove
from os.path import exists, join
import glob
import matplotlib
# matplotlib.use('Agg')
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D, Poly3DCollection, Line3DCollection
from descartes import PolygonPatch

import matplotlib.pyplot as plt  # matplotlib.use('Agg') # TkAgg
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

import __init__
from global_info import global_info
def breakpoint():
    import pdb; pdb.set_trace()

def bp():
    import pdb; pdb.set_trace()

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

Dmap = {'c': 0, 'o': 1} # ''
Mmap = {'p++': 0, 'se3': 1}
Tmap = {'N': 0, 'R': 1, 'T': 2 , 'S': 3}
exp_dict = {}
exp_dict[(0, 0, 0)] = 2.4074
exp_dict[(0, 1, 0)] = 2.40941
exp_dict[(0, 1, 1)] = 2.4058
exp_dict[(0, 1, 2)] = 2.406971

def get_txt(fn):
    point_normal_set = np.loadtxt(fn, delimiter=' ').astype(np.float32)
    return point_normal_set

def plot_arrows(fn=None):
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

def check_rts(file_name):
    all_rts   = np.load(file_name, allow_pickle=True).item()
    # evaluate per category as well
    xyz_err_dict = {}
    rpy_err_dict = {}
    keys_dict    = {}
    bad_instances = ['3b26c9021d9e31a7ad8912880b776dcf', '9a70dd116325eb2df6b14e57dce52ee1', '591153135f8571a69fc36bc06f1db2fa', '8d861e4406a9d3258534bca5562e21be_02876657', 'a95db9f4cb692657a09c6a60c6dc420', '3ae3a9b74f96fef28fe15648f042f0d9', ]
    for key, value_dict in all_rts.items():
        category_name = key.split('_')[-1]
        instance_name = key.split('_')[-2]
        if instance_name in bad_instances:
            continue
        if category_name not in xyz_err_dict:
            xyz_err_dict[category_name] = []
            rpy_err_dict[category_name] = []
            keys_dict[category_name] = []
        if isinstance(value_dict['rpy_err']['baseline'], list):
            rpy_err_dict[category_name].append(value_dict['rpy_err']['baseline'][0])
        else:
            rpy_err_dict[category_name].append(value_dict['rpy_err']['baseline'])
        if isinstance(value_dict['xyz_err']['baseline'], list):
            xyz_err_dict[category_name].append(value_dict['xyz_err']['baseline'][0])
        else:
            xyz_err_dict[category_name].append(value_dict['xyz_err']['baseline'])
        keys_dict[category_name].append(key)
    all_categorys = xyz_err_dict.keys()

    print('category\trotation error\ttranslation error')
    for category in all_categorys:
        print(f'{categories_id[category]}\t{np.array(rpy_err_dict[category]).mean():0.4f}\t{np.array(xyz_err_dict[category]).mean():0.4f}')
        all_bad_indexs = np.where(np.array(rpy_err_dict[category]) > 20)[0]
        #
        # all_bad_indexs = np.where(np.array(xyz_err_dict[category]) > 5)[0]
        # for index in all_bad_indexs:
        #     # index = rpy_err_dict[category_name].index(max(rpy_err_dict[category_name]))
        #     basename = keys_dict[category_name][index]
        #     print(f'--problem data is ', basename, 'R: ', rpy_err_dict[category][index], 'T: ', xyz_err_dict[category][index])
        #     input = all_rts[basename]['in']
            # try:
            #     gt   = all_rts[basename]['gt']
            #     pred = all_rts[basename]['pred']
            #     plot3d_pts([[input], [pred], [gt]], [['input'], ['pred nocs'], ['gt nocs']],  title_name=['0', '1', '2'], s=4**2, dpi=300, axis_off=False, color_channel=[[gt], [gt], [gt]])
            # except:
            #     plot3d_pts([[input]], [['input']],  title_name=['0'], s=4**2, dpi=300, axis_off=False)
    # 5 degrees accuracy
    num_parts = 1
    print('For {} object, {}, 5 degrees accuracy is: '.format(cfg.domain, cfg.nocs))
    for category in all_categorys:
        r_err = np.array(rpy_err_dict[category])
        num_valid = r_err.shape[0]
        r_acc = []
        for j in range(num_parts):
            idx = np.where(r_err < 5)[0]
            acc   = len(idx) / num_valid
            r_acc.append(acc)
        print(categories_id[category], " ".join(["{:0.4f}".format(x) for x in r_acc]))
    print('\n')
    # 5 degrees & 0.05
    print('For {} object, {}, 5 degrees, 5 cms accuracy is: '.format(cfg.domain, cfg.nocs))
    for category in all_categorys:
        num_valid = r_err.shape[0]
        rt_acc = []
        t_err  = np.array(xyz_err_dict[category])
        for j in range(num_parts):
            idx = np.where(r_err < 5)[0]
            acc   = len(np.where( t_err[idx] < 0.05 )[0]) / num_valid
            rt_acc.append(acc) # two modes
        print(categories_id[category], " ".join(["{:0.4f}".format(x) for x in rt_acc]))
    print('\n')
    plot_distribution(rpy_err_dict[category_name], labelx='r_err', labely='frequency', title_name='rotation_error', sub_name=cfg.exp_num, save_fig=True)
    plot_distribution(xyz_err_dict[category_name], labelx='t_err', labely='frequency', title_name='translation_error', sub_name=cfg.exp_num, save_fig=True)

# 1. viz rts
def get_rts_filename(cfg):
    #
    key1, key2, key3 = cfg.words.split(',')
    exp_name  = exp_dict[(Dmap[key1], Mmap[key2], Tmap[key3])]
    print('using ', cfg.key_words, exp_name)
    file_name = f'{second_path}/results/test_pred/obman/{exp_name}_unseen_part_rt_pn_general.npy'
    return file_name

import joblib
def main():
        split = 'test'
        fpath  = '/home/dragon/Documents/external/modelnet40'
        f_train= f'{fpath}/airplane_{split}_2048.pk'
        with open(f_train, "rb") as f:
            full_data = joblib.load(f)
        for i in [0, 1, 2, 13, 16, 17, 112, 130, 135, 145, 212, 221, 305, 308]:
            input = full_data[i]
            plot3d_pts([[input]], [[f'{i}th']], title_name=['input'], s=3**2, dpi=300, axis_off=False, show_fig=True)


if __name__ == '__main__':
    main()
