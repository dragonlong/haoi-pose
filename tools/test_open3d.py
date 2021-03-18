import open3d as o3d
import numpy as np
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt  # matplotlib.use('Agg') # TkAgg
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from sklearn.externals import joblib


def rotate_about_axis(theta, axis='x'):
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, cos(theta), -sin(theta)],
                      [0, sin(theta), cos(theta)]])

    elif axis == 'y':
        R = np.array([[cos(theta), 0, sin(theta)],
                      [0, 1, 0],
                      [-sin(theta), 0, cos(theta)]])

    elif axis == 'z':
        R = np.array([[cos(theta), -sin(theta), 0],
                      [sin(theta), cos(theta),  0],
                      [0, 0, 1]])
    return R

def bp():
    import pdb;pdb.set_trace()

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

def plot3d_pts(pts, pts_name, s=1, dpi=350, title_name=None, sub_name='default', arrows=None, \
                    color_channel=None, colorbar=False, limits=None,\
                    bcm=None, puttext=None, view_angle=None,\
                    save_fig=False, save_path=None, flip=True,\
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
    all_poss=['o', 'o', 'o', 'o','o', '*', '.','o', 'v','^','>','<','s','p','*','h','H','D','d','1','','']
    c_set   = ['r', 'b', 'g', 'k', 'm']
    num     = len(pts)
    for m in range(num):
        ax = plt.subplot(1, num, m+1, projection='3d')
        if view_angle==None:
            ax.view_init(elev=36, azim=-49)
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
                offset_sub = offset_sub * 0.2
                if len(points.shape) < 2:
                    points = points.reshape(-1, 3)
                if len(offset_sub.shape) < 2:
                    offset_sub = offset_sub.reshape(-1, 3)
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
        if save_path is None:
            if not os.path.exists('./results/test/'):
                os.makedirs('./results/test/')
            fig.savefig('./results/test/{}_{}.png'.format(sub_name, title_name[0]), pad_inches=0)
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig('{}/{}_{}.png'.format(save_path, sub_name, title_name[0]), pad_inches=0)
    if mode != 'continuous':
        plt.close()

# https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/
if __name__ == '__main__':
    # xyz = np.random.rand(512, 3)
    fn  = '/home/dragon/Documents/ICML2021/data/modelnet40/airplane_0002.txt'
    xyz = np.loadtxt(fn, delimiter=',').astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(xyz[:, 3:])

    fpath  = '/home/dragon/Documents/external/modelnet40/'
    f_train= f'{fpath}/airplane_train_2048.pk'
    f_test = f'{fpath}/airplane_test_2048.pk'

    with open(f_train, "rb") as f:
       train_pts = joblib.load(f)

    with open(f_test, "rb") as obj_f:
        test_pts  = joblib.load(obj_f)

    print(train_pts.shape, test_pts.shape)
    # for j in range(0, 1000, 3):
    #     xyz = train_pts[j]
    #     p1  = train_pts[j+1]
    #     p2  = train_pts[j+2]
    #     plot3d_pts([[xyz], [p1], [p2]], [['input'], ['projection'], ['p']], title_name=['complete', 'visible', 'cam'], s=0.2)

    nx, ny, nz = 5, 5, 5
    for m in range(train_pts.shape[0]): #
        xyz = train_pts[m]
        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(xyz[:, :3])
        visible_dict = {}
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    theta_x = 360/8 * i
                    Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
                    theta_y = 360/8 * j
                    Ry = rotate_about_axis(theta_y / 180 * np.pi, axis='y')
                    theta_z = 360/8 * k
                    Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
                    r = np.matmul(Ry, Rx).astype(np.float32)
                    r = np.matmul(Rz, r).astype(np.float32)

                    camera_location, radius = np.matmul(np.array([[2, 2, 2]]), r.T), 1000 # from randomly rotate
                    camera_location = camera_location.astype(np.float64).reshape(3, 1)
                    visible_pts = pcd.hidden_point_removal(camera_location, radius)
                    # p1 = xyz[visible_pts[1]][:, :3]
                    visible_dict[f'{i}_{j}_{k}'] = visible_pts[1]
                    # p2 = np.matmul(p1, r)
                    # if m < 20:
                    #     save_name = f'/home/dragon/Documents/external/modelnet40/train/{m}_{i}_{j}_{k}.txt'
                    #     print('saving into ', save_name)
                    #     np.savetxt(save_name, p1)
                    # save_name = f'/home/dragon/Documents/ICML2021/data/modelnet40/airplane_r/{i}_{j}_{k}.txt'
                    # print('saving into ', save_name)
                    # np.savetxt(save_name, p2)
                    # print(p1.shape)
                    # plot3d_pts([[xyz], [p1], [p2]], [['input'], ['projection'], ['p']], title_name=['complete', 'visible', 'cam'], s=0.2)
        save_name = f'/home/dragon/Documents/external/modelnet40/train/{m:04d}.npy'
        np.save(save_name, arr=visible_dict)
        print('saving to ', save_name)

    nx, ny, nz = 5, 5, 5
    for m in range(test_pts.shape[0]): #
        xyz = test_pts[m]
        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(xyz[:, :3])
        visible_dict = {}
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    theta_x = 360/8 * i
                    Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
                    theta_y = 360/8 * j
                    Ry = rotate_about_axis(theta_y / 180 * np.pi, axis='y')
                    theta_z = 360/8 * k
                    Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
                    r = np.matmul(Ry, Rx).astype(np.float32)
                    r = np.matmul(Rz, r).astype(np.float32)

                    camera_location, radius = np.matmul(np.array([[2, 2, 2]]), r.T), 1000 # from randomly rotate
                    camera_location = camera_location.astype(np.float64).reshape(3, 1)
                    visible_pts = pcd.hidden_point_removal(camera_location, radius)
                    # p1 = xyz[visible_pts[1]][:, :3]
                    visible_dict[f'{i}_{j}_{k}'] = visible_pts[1]
                    # p2 = np.matmul(p1, r)
                    # if m < 20:
                    #     save_name = f'/home/dragon/Documents/external/modelnet40/train/{m}_{i}_{j}_{k}.txt'
                    #     print('saving into ', save_name)
                    #     np.savetxt(save_name, p1)
                    # save_name = f'/home/dragon/Documents/ICML2021/data/modelnet40/airplane_r/{i}_{j}_{k}.txt'
                    # print('saving into ', save_name)
                    # np.savetxt(save_name, p2)
                    # print(p1.shape)
                    # plot3d_pts([[xyz], [p1], [p2]], [['input'], ['projection'], ['p']], title_name=['complete', 'visible', 'cam'], s=0.2)
        save_name = f'/home/dragon/Documents/external/modelnet40/test/{m:04d}.npy'
        np.save(save_name, arr=visible_dict)
        print('saving to ', save_name)
