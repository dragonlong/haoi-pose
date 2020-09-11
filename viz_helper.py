import os
import yaml
import numpy as np
import time
import h5py

from random import randint, sample
import matplotlib
import matplotlib.pyplot as plt  #
from voxelvis import PointVis

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
    my_dir = '/home/dragon/Dropbox/cvpr2021/viz/kpconv/temporal_anchors/'
    filenames = os.listdir(my_dir)
    for i in range(len(filenames)):
        filename = filenames[i]
        viz_file = f'{my_dir}/{filename}'
        print(f'Now checking {viz_file}')
        gt_data_handle  = np.load(viz_file, allow_pickle=True)
        gt_dict         = gt_data_handle.item()
        #
        for key, value in gt_dict.items():
            try:
                print(key, value.shape)
            except:
                print(key, value)
        vis = PointVis(target_pts=None, viz_dict=gt_dict)
        vis.run()

