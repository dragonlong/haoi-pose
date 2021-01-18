import numpy as np
import os
import os
import yaml
import numpy as np
import time
import h5py

from random import randint, sample
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common.vis_utils import plot3d_pts

def visualize_pointcloud(points, normals=None, title_name='0',
                         out_file=None, show=False):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_zlim(0, 1.0)
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if title_name is not None:
        plt.title(title_name)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

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

if __name__ == '__main__':
    my_dir      = 'outputs/media/full_viz'
    filenames = [f'{my_dir}/{filename}' for filename in os.listdir(my_dir)]
    # print(filenames)
    for i, filename in enumerate(filenames):
        data = np.load(filename, allow_pickle=True).item()
        for key, value in data.items():
            try:
                data[key] = value.numpy()
            except:
                print(key, value)
        # 'points', 'points.occ', 'inputs', 'inputs.normals'
        for j in range(data['points'].shape[0]):
            plot3d_pts([[data['inputs'][j]], [data['points'][j]]], [['NOCS input'], ['Occupancy Label']], s=2, dpi=350, title_name=[f'{j}th', f'{j}th'], sub_name='default', color_channel=[[data['inputs'][j]], [ data['points.occ'][j].reshape(-1, 1) * np.array([[255, 0, 0]]) ]])
            # visualize_pointcloud(data['inputs'][j], title_name='inputs', show=True)
            # visualize_pointcloud(data['points'][j], title_name='occ pts', show=True)
