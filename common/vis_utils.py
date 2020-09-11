import matplotlib
# matplotlib.use('Agg')
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.collections import PatchCollection

import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from descartes import PolygonPatch

import matplotlib.pyplot as plt  # matplotlib.use('Agg') # TkAgg
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

import os
import cv2
import vispy
from vispy.scene import visuals, SceneCanvas

# Colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# use vispy(opengl) to visualize large scale point cloud efficiently
class tiny_canvas():
    def __init__(self, semantics=True, instances=False):
        self.semantics = semantics
        self.instances = instances
        self.reset()

    def reset(self):
        """ Reset. """
        # create canvas
        # create viz window
        # link to data
        # update points & color labels
        # close
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0, 0)
        self.scan_vis = visuals.Markers()
        self.scan_view.camera = 'turntable'
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)
        # add semantics
        if self.semantics:
          print("Using semantics in visualizer")
          self.sem_view = vispy.scene.widgets.ViewBox(
              border_color='white', parent=self.canvas.scene)
          self.grid.add_widget(self.sem_view, 0, 1)
          self.sem_vis = visuals.Markers()
          self.sem_view.camera = 'turntable'
          self.sem_view.add(self.sem_vis)
          visuals.XYZAxis(parent=self.sem_view.scene)
          # self.sem_view.camera.link(self.scan_view.camera)

        if self.instances:
          print("Using instances in visualizer")
          self.inst_view = vispy.scene.widgets.ViewBox(
              border_color='white', parent=self.canvas.scene)
          self.grid.add_widget(self.inst_view, 0, 2)
          self.inst_vis = visuals.Markers()
          self.inst_view.camera = 'turntable'
          self.inst_view.add(self.inst_vis)
          visuals.XYZAxis(parent=self.inst_view.scene)
          # self.inst_view.camera.link(self.scan_view.camera)

        # img canvas size
        self.multiplier = 1
        self.canvas_W = 1024
        self.canvas_H = 64
        if self.semantics:
          self.multiplier += 1
        if self.instances:
          self.multiplier += 1

        # new canvas for img
        self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                      size=(self.canvas_W, self.canvas_H * self.multiplier))
        # grid
        self.img_grid = self.img_canvas.central_widget.add_grid()
        # interface (n next, b back, q quit, very simple)
        self.img_canvas.events.key_press.connect(self.key_press)
        self.img_canvas.events.draw.connect(self.draw)

        # add a view for the depth
        self.img_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.img_canvas.scene)
        self.img_grid.add_widget(self.img_view, 0, 0)
        self.img_vis = visuals.Image(cmap='viridis')
        self.img_view.add(self.img_vis)

        # add semantics
        if self.semantics:
          self.sem_img_view = vispy.scene.widgets.ViewBox(
              border_color='white', parent=self.img_canvas.scene)
          self.img_grid.add_widget(self.sem_img_view, 1, 0)
          self.sem_img_vis = visuals.Image(cmap='viridis')
          self.sem_img_view.add(self.sem_img_vis)

        # add instances
        if self.instances:
          self.inst_img_view = vispy.scene.widgets.ViewBox(
              border_color='white', parent=self.img_canvas.scene)
          self.img_grid.add_widget(self.inst_img_view, 2, 0)
          self.inst_img_vis = visuals.Image(cmap='viridis')
          self.inst_img_view.add(self.inst_img_vis)

    def viz(self, point_cloud, color_labels, face_color):
        # first open data
        self.scan.open_scan(self.scan_names[self.offset])
        if self.semantics:
          self.scan.open_label(self.label_names[self.offset])
          self.scan.colorize()

        # then change names
        title = "scan " + str(self.offset)
        self.canvas.title = title
        self.img_canvas.title = title

        # then do all the point cloud stuff

        # plot scan
        power = 16
        # print()
        range_data = np.copy(self.scan.unproj_range)
        # print(range_data.max(), range_data.min())
        range_data = range_data**(1 / power)
        # print(range_data.max(), range_data.min())
        viridis_range = ((range_data - range_data.min()) /
                         (range_data.max() - range_data.min()) *
                         255).astype(np.uint8)
        viridis_map = self.get_mpl_colormap("viridis")
        viridis_colors = viridis_map[viridis_range]
        self.scan_vis.set_data(self.scan.points,
                               face_color=viridis_colors[..., ::-1],
                               edge_color=viridis_colors[..., ::-1],
                               size=1)

        # plot semantics
        if self.semantics:
          self.sem_vis.set_data(self.scan.points,
                                face_color=self.scan.sem_label_color[..., ::-1],
                                edge_color=self.scan.sem_label_color[..., ::-1],
                                size=1)

        # plot instances
        if self.instances:
          self.inst_vis.set_data(self.scan.points,
                                 face_color=self.scan.inst_label_color[..., ::-1],
                                 edge_color=self.scan.inst_label_color[..., ::-1],
                                 size=1)

        # now do all the range image stuff
        # plot range image
        data = np.copy(self.scan.proj_range)
        # print(data[data > 0].max(), data[data > 0].min())
        data[data > 0] = data[data > 0]**(1 / power)
        data[data < 0] = data[data > 0].min()
        # print(data.max(), data.min())
        data = (data - data[data > 0].min()) / \
            (data.max() - data[data > 0].min())
        # print(data.max(), data.min())
        self.img_vis.set_data(data)
        self.img_vis.update()

        if self.semantics:
          self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
          self.sem_img_vis.update()

        if self.instances:
          self.inst_img_vis.set_data(self.scan.proj_inst_color[..., ::-1])
          self.inst_img_vis.update()

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
          self.canvas.events.key_press.unblock()
        if self.img_canvas.events.key_press.blocked():
          self.img_canvas.events.key_press.unblock()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        self.img_canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()

def viz_voxels():
    # use openGL to visualize voxels
    pass


def plot_scene_w_grasps(list_obj_verts, list_obj_faces, list_obj_handverts, list_obj_handfaces, plane_parameters):
    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins

    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    # We will convert this into a single mesh, and then use add_group_meshs to plot it in 3D
    allverts = np.zeros((0,3))
    allfaces = np.zeros((0,3))
    colors = []
    for i in range(len(list_obj_verts)):
        allfaces = np.concatenate((allfaces, list_obj_faces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_verts[i]))
        colors = np.concatenate((colors, ['r']*len(list_obj_faces[i])))

    for i in range(len(list_obj_handverts)):
        allfaces = np.concatenate((allfaces, list_obj_handfaces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_handverts[i]))
        colors = np.concatenate((colors, ['b']*len(list_obj_handfaces[i])))

    allfaces = np.int32(allfaces)
    print(np.max(allfaces))
    print(np.shape(allverts))
    add_group_meshs(ax, allverts, allfaces, alpha=1, c=colors)

    cam_equal_aspect_3d(ax, np.concatenate(list_obj_verts, 0), flip_z=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    # Show plane too:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    step = 0.05
    border = 0.0 #step
    X, Y = np.meshgrid(np.arange(xlim[0]-border, xlim[1]+border, step),
               np.arange(ylim[0]-border, ylim[1]+border, step))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
      for c in range(X.shape[1]):
        Z[r, c] = (-plane_parameters[0] * X[r, c] - plane_parameters[1] * Y[r, c] + plane_parameters[3])/plane_parameters[2]
    ax.plot_wireframe(X, Y, Z, color='r')

    pressed_keyboard = False
    while not pressed_keyboard:
        pressed_keyboard = plt.waitforbuttonpress()

    plt.close(fig)


def plot_hand_w_object(obj_verts, obj_faces, hand_verts, hand_faces, pts=None, jts=None, nmls=None, viz_m=False, viz_c=False, viz_j=False, viz_n=False, save_path=None, flip=True, save=False):
    """
    Functions taken from the ObMan dataset repo (https://github.com/hassony2/obman)
    """
    colors = ['r']*len(hand_faces) + ['b']*len(obj_faces)

    frames = []
    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(111, projection='3d')
    # ax.axis('off')
    # plt.show('off')

    verts = hand_verts
    add_group_meshs(ax, np.concatenate((verts, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, obj_verts, flip_x=flip, flip_y=flip)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    pressed_keyboard = False
    while not pressed_keyboard:
        pressed_keyboard = plt.waitforbuttonpress()
    if save:
        fig.savefig(save_path, pad_inches=0)
    plt.close(fig)

    return

def add_mesh(ax, verts, faces, alpha=0.1, c='b'):
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == 'b':
        face_color = (141 / 255, 184 / 255, 226 / 255)
    elif c == 'r':
        face_color = (226 / 255, 184 / 255, 141 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

def add_group_meshs(ax, verts, faces, alpha=0.1, c='b'):
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = []
    for i in range(len(c)):
        if c[i] == 'b':
            face_color.append((141 / 255, 184 / 255, 226 / 255))
        elif c[i] == 'r':
            face_color.append((226 / 255, 184 / 255, 141 / 255))
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)


def cam_equal_aspect_3d(ax, verts, flip_x=False, flip_y=False, flip_z=False):
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
    if flip_y:
        ax.set_ylim(centers[1] + r, centers[1] - r)
    else:
        ax.set_ylim(centers[1] - r, centers[1] + r)

    if flip_z:
        ax.set_zlim(centers[2] + r, centers[2] - r)
    else:
        ax.set_zlim(centers[2] - r, centers[2] + r)



def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)


def make_log_img(depth, mask, pred, gt, color_fn):
    # input should be [depth, pred, gt]
    # make range image (normalized to 0,1 for saving)
    depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
    out_img = cv2.applyColorMap(
        depth, get_mpl_colormap('viridis')) * mask[..., None]

    # make label prediction
    pred_color = color_fn((pred * mask).astype(np.int32))
    out_img = np.concatenate([out_img, pred_color], axis=0)
    # make label gt
    gt_color = color_fn(gt)
    out_img = np.concatenate([out_img, gt_color], axis=0)
    return (out_img).astype(np.uint8)

def save_to_log(logdir, logger, info, epoch, img_summary=False, imgs=[]):
    # save scalars
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    if img_summary and len(imgs) > 0:
        directory = os.path.join(logdir, "predictions")
        if not os.path.isdir(directory):
            os.makedirs(directory)
        for i, img in enumerate(imgs):
            name = os.path.join(directory, str(i) + ".png")
            cv2.imwrite(name, img)

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

def plot2d_img(imgs, title_name=None, dpi=200, cmap=None, save_fig=False, show_fig=False, save_path=None, sub_name='0'):
    # fig     = plt.figure(dpi=dpi)
    # step = 0
    # heights = [50 for a in range(1)]
    # widths = [60 for a in range(2)]
    # cmaps = [['viridis', 'binary'], ['plasma', 'coolwarm'], ['Greens', 'copper']]
    # fig_width = 10  # inches
    # fig_height = fig_width * sum(heights) / sum(widths)
    # fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios':heights}, dpi=dpi) #define to be 2 rows, and 4cols.

    # for i in range(1):
    #     for j in range(2):
    #         axes[j].imshow(imgs[j])
    #         axes[j].axis('off')
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    #             hspace = 0.01, wspace = 0.01)
    all_poss=['.','o','v','^']
    num     = len(imgs)
    for m in range(num):
        ax = plt.subplot(1, num, m+1)
        if cmap is None:
            plt.imshow(imgs[m])
        else:
            plt.imshow(imgs[m], cmap=cmap)
        plt.title(title_name[m])
        plt.axis('off')
        plt.grid('off')
    if show_fig:
        plt.show()
    if save_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig('{}/{}.png'.format(save_path, sub_name, pad_inches=0))

def plot3d_pts(pts, pts_name, s=2, dpi=150, title_name=None, sub_name='default', \
                    color_channel=None, colorbar=False, limits=None,\
                    bcm=None, puttext=None, view_angle=None,\
                    save_fig=False, save_path=None, \
                    axis_off=False, show_fig=True):
    """
    fig using
    """
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    top     = plt.cm.get_cmap('Oranges_r', 128)
    bottom  = plt.cm.get_cmap('Blues', 128)

    colors = np.vstack((top(np.linspace(0, 1, 10)),
                           bottom(np.linspace(0, 1, 10))))
    # colors = ListedColormap(newcolors, name='OrangeBlue')
    # colors  = cmap(np.linspace(0., 1., 5))
    # colors = ['Blues', 'Blues',  'Blues', 'Blues', 'Blues']
    all_poss=['o', 'o','o', '.','o', '*', '.','o', 'v','^','>','<','s','p','*','h','H','D','d','1','','']
    num     = len(pts)
    for m in range(num):
        ax = plt.subplot(1, num, m+1, projection='3d')
        if view_angle==None:
            ax.view_init(elev=36, azim=-49)
        else:
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
        if len(pts[m]) > 1:
            for n in range(len(pts[m])):
                if n==len(pts[m]) - 1:
                    s = 8**2
                if color_channel is None:
                    ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2], marker=all_poss[n], s=s, cmap=colors[n], label=pts_name[m][n])
                else:
                    if colorbar:
                        rgb_encoded = color_channel[m][n]
                    else:
                        rgb_encoded = (color_channel[m][n] - np.amin(color_channel[m][n], axis=0, keepdims=True))/np.array(np.amax(color_channel[m][n], axis=0, keepdims=True) - np.amin(color_channel[m][n], axis=0, keepdims=True))
                    if len(pts[m])==3 and n==2:
                        p = ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2],  marker=all_poss[4], s=s, c=rgb_encoded, label=pts_name[m][n])
                    else:
                        p = ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2],  marker=all_poss[n], s=s, c=rgb_encoded, label=pts_name[m][n])
                    if colorbar:
                        fig.colorbar(p)
        else:
            for n in range(len(pts[m])):
                if color_channel is None:
                    p = ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2], marker=all_poss[n], s=s, cmap=colors[n])
                else:
                    if colorbar:
                        rgb_encoded = color_channel[m][n]
                    else:
                        rgb_encoded = (color_channel[m][n] - np.amin(color_channel[m][n], axis=0, keepdims=True))/np.array(np.amax(color_channel[m][n], axis=0, keepdims=True) - np.amin(color_channel[m][n], axis=0, keepdims=True))
                    if len(pts[m])==3 and n==2:
                        p = ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2],  marker=all_poss[4], s=s, c=rgb_encoded)
                    else:
                        p = ax.scatter(pts[m][n][:, 0],  pts[m][n][:, 1], pts[m][n][:, 2],  marker=all_poss[n], s=s, c=rgb_encoded)
                    if colorbar:
                        # fig.colorbar(p)
                        fig.colorbar(p, ax=ax)

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
        if limits is None:
            limits = [[-1, 1], [-1, 1], [-1, 1]]
        set_axes_equal(ax, limits=limits)
    if show_fig:
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
    plt.close()

def plot_imgs(imgs, imgs_name, title_name='default', sub_name='default', save_path=None, save_fig=False, axis_off=False, show_fig=True, dpi=150):
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    num = len(imgs)
    for m in range(num):
        ax1 = plt.subplot(1, num, m+1)
        plt.imshow(imgs[m].astype(np.uint8))
        if title_name is not None:
            plt.title(title_name[0]+ ' ' + imgs_name[m])
        else:
            plt.title(imgs_name[m])
    if show_fig:
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

    plt.close()

def plot_arrows(points, offset=None, joint=None, whole_pts=None, title_name='default', limits=None, idx=None, dpi=200, s=0.1, thres_r=0.1, show_fig=True, sparse=True, index=0, save=False, save_path=None):
    """
    points: [N, 3]
    offset: [N, 3] or list of [N, 3]
    joint : [P0, ll], a list, array

    """
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    colors  = cmap(np.linspace(0., 1., 5))
    c_set   = ['r', 'b', 'g', 'k', 'm']
    all_poss=['.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    num     = len(points)
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=60, azim=150)
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[0], s=s)
    if whole_pts is not None:
        p = ax.scatter(whole_pts[:, 0], whole_pts[:, 1], whole_pts[:, 2],  marker=all_poss[1], s=s)
    if not isinstance(offset, list):
        offset = [offset]

    for j, offset_sub in enumerate(offset):
        if sparse:
            if idx is None:
                ax.quiver(points[::10, 0], points[::10, 1], points[::10, 2], offset_sub[::10, 0], offset_sub[::10, 1], offset_sub[::10, 2], color=c_set[j])
            else:
                points = points[idx, :]
                ax.quiver(points[::2, 0], points[::2, 1], points[::2, 2], offset_sub[::2, 0], offset_sub[::2, 1], offset_sub[::2, 2], color=c_set[j])
        else:
            if idx is None:
                ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset_sub[:, 0], offset_sub[:, 1], offset_sub[:, 2], color=c_set[j])
            else:
                print(idx)
                ax.quiver(points[idx[:], 0], points[idx[:], 1], points[idx[:], 2], offset_sub[idx[:], 0], offset_sub[idx[:], 1], offset_sub[idx[:], 2], color=c_set[j])
    if joint is not None:
        for sub_j in joint:
            length = 0.5
            sub_j[0] = sub_j[0].reshape(1,3)
            sub_j[1] = sub_j[1].reshape(-1)
            ax.plot3D([sub_j[0][0, 0]- length * sub_j[1][0], sub_j[0][0, 0] + length * sub_j[1][0]], \
                      [sub_j[0][0, 1]- length * sub_j[1][1], sub_j[0][0, 1] + length * sub_j[1][1]], \
                      [sub_j[0][0, 2]- length * sub_j[1][2], sub_j[0][0, 2] + length * sub_j[1][2]],  'blue', linewidth=8)
    if limits is None:
        limits = [[-1, 1], [-1, 1], [-1, 1]]
    set_axes_equal(ax, limits=limits)
    plt.title(title_name)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    if show_fig:
        plt.show()

    if save:
        if save_path is None:
            if not os.path.exists('./results/test/'):
                os.makedirs('./results/test/')
            fig.savefig('./results/test/{}_{}.png'.format(index, title_name[0]), pad_inches=0)
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig('{}/{}_{}.png'.format(save_path, index, title_name[0]), pad_inches=0)
    plt.close()

def plot_lines(orient_vect):
    """
    orient_vect: list of [3] or None
    """
    fig     = plt.figure(dpi=150)
    cmap    = plt.cm.jet
    colors  = cmap(np.linspace(0., 1., 5))
    c_set   = ['r', 'b', 'g', 'k', 'm']
    all_poss=['.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=32, azim=-54)
    for sub_j in orient_vect:
        if sub_j is not None:
            length = 0.5
            ax.plot3D([0, sub_j[0]], \
                      [0, sub_j[1]], \
                      [0, sub_j[2]],  'blue', linewidth=5)
    plt.show()
    plt.close()

def plot_arrows_list(points_list, offset_list, whole_pts=None, title_name='default', limits=None, dpi=200, s=5, lw=1, length=0.5, view_angle=None, sparse=True, axis_off=False):
    """
    points: list of [N, 3]
    offset: nested list of [N, 3]
    joint : [P0, ll], 2-order nested list, array

    """
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    colors  = cmap(np.linspace(0., 1., 5))
    c_set = ['r', 'g', 'b', 'k', 'm']
    all_poss=['.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    ax = plt.subplot(1, 1, 1, projection='3d')
    if view_angle==None:
        ax.view_init(elev=36, azim=-49)
    else:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    if whole_pts is not None:
        p = ax.scatter(whole_pts[:, 0], whole_pts[:, 1], whole_pts[:, 2],  marker=all_poss[0], s=s)
    for i in range(len(points_list)):
        points = points_list[i]
        p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[1], s=10,  cmap=colors[i+1])
        offset = offset_list[i]
        if sparse:
            ls =5
            ax.quiver(points[::ls, 0], points[::ls, 1], points[::ls, 2], offset[::ls, 0], offset[::ls, 1], offset[::ls, 2], color=c_set[i], linewidth=lw)
        else:
            ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset[:, 0], offset[:, 1], offset[:, 2], color='r', linewidth=lw)
    if limits is None:
        limits = []
    set_axes_equal(ax, limits=limits)
    plt.title(title_name)
    if axis_off:
        plt.axis('off')
        plt.grid('off')
    plt.show()
    plt.close()

def plot_joints_bb_list(points_list, offset_list=None, joint_list=None, whole_pts=None, bcm=None, view_angle=None, title_name='default', sub_name='0', dpi=200, s=15, lw=1, length=0.5, sparse=True, save_path=None, show_fig=True, save_fig=False):
    """
    points: list of [N, 3]
    offset: nested list of [N, 3]
    joint : [P0, ll], 2-order nested list, array

    """
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    top     = plt.cm.get_cmap('Oranges_r', 128)
    bottom  = plt.cm.get_cmap('Blues', 128)

    colors = np.vstack((top(np.linspace(0, 1, 10)),
                           bottom(np.linspace(0, 1, 10))))
    c_set = ['g', 'b', 'm', 'y', 'r', 'c']
    all_poss=['.','o','.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    ax = plt.subplot(1, 1, 1, projection='3d')
    if view_angle==None:
        ax.view_init(elev=36, azim=-49)
    else:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    # ax.view_init(elev=46, azim=-164)
    pts_name = ['part {}'.format(j) for j in range(10)]
    if whole_pts is not None:
        for m, points in enumerate(whole_pts):
            p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[1], s=s, cmap=colors[m], label=pts_name[m])
    center_pt = np.mean(whole_pts[0], axis=0)
    for i in range(len(points_list)):
        points = points_list[i]
        # p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[i], s=s,  c='c')
        if offset_list is not None:
            offset = offset_list[i]# with m previously
            if sparse:
                ax.quiver(points[::50, 0], points[::50, 1], points[::50, 2], offset[::50, 0], offset[::50, 1], offset[::50, 2], color=c_set[i])
            else:
                ax.quiver(points[:, 0], points[:, 1], points[:, 2], offset[:, 0], offset[:, 1], offset[:, 2], color='r')
        # we have two layers
        palette = get_tableau_palette()
        if joint_list is not None:
            if joint_list[i] is not []:
                joint  = joint_list[i] # [[1, 3], [1, 3]]
                jp = joint['p'].reshape(-1)
                jp[1] = 0.5
                jl = joint['l'].reshape(-1)
                # mean
                ax.quiver(jp[0]- 0*length * abs(jl[0]), jp[1]- 0*length * abs(jl[1]), jp[2]- 0*length * abs(jl[2]),\
                        1* length * jl[0], 1 * length * jl[1], 1* length * jl[2], color='black', linewidth=3, arrow_length_ratio=0.3)
                # ax.quiver(jp[0]- 0*length * abs(jl[0]), jp[1]- 0*length * abs(jl[1]), jp[2]- 0*length * abs(jl[2]),\
                #         1* length * jl[0], 1 * length * jl[1], 1* length * jl[2], cmap=palette[0], linewidth=3, arrow_length_ratio=0.3)
    # ax.dist = 8
    print('viewing distance is ', ax.dist)
    if bcm is not None:
        for j in range(len(bcm)):
            color_s = 'gray'
            lw_s =1.5
            # if j == 1:
            #     color_s = 'red'
            #     lw_s = 2
            ax.plot3D([bcm[j][0][0], bcm[j][2][0], bcm[j][6][0], bcm[j][4][0], bcm[j][0][0]], \
                [bcm[j][0][1], bcm[j][2][1], bcm[j][6][1], bcm[j][4][1], bcm[j][0][1]], \
                [bcm[j][0][2], bcm[j][2][2], bcm[j][6][2], bcm[j][4][2], bcm[j][0][2]], color=color_s, linewidth=lw_s)

            ax.plot3D([bcm[j][1][0], bcm[j][3][0], bcm[j][7][0], bcm[j][5][0], bcm[j][1][0]], \
                [bcm[j][1][1], bcm[j][3][1], bcm[j][7][1], bcm[j][5][1], bcm[j][1][1]], \
                [bcm[j][1][2], bcm[j][3][2], bcm[j][7][2], bcm[j][5][2], bcm[j][1][2]], color=color_s, linewidth=lw_s)

            for pair in [[0, 1], [2, 3], [4, 5], [6, 7]]:
                ax.plot3D([bcm[j][pair[0]][0], bcm[j][pair[1]][0]], \
                    [bcm[j][pair[0]][1], bcm[j][pair[1]][1]], \
                    [bcm[j][pair[0]][2], bcm[j][pair[1]][2]], color=color_s, linewidth=lw_s)

    plt.title(title_name, fontsize=10)
    plt.axis('off')
    # plt.legend('off')
    plt.grid('off')
    limits = [[0, 1], [0, 1], [0, 1]]
    set_axes_equal(ax, limits)
    if show_fig:
        plt.show()
    if save_fig:
        if save_path is None:
            if not os.path.exists('./results/test/'):
                os.makedirs('./results/test/')
            fig.savefig('./results/test/{}_{}.png'.format(sub_name, title_name), pad_inches=0)
            print('saving figure into ', './results/test/{}_{}.png'.format(sub_name, title_name))
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig('{}/{}_{}.png'.format(save_path, sub_name, title_name), pad_inches=0)
            print('saving fig into ', '{}/{}_{}.png'.format(save_path, sub_name, title_name))
    plt.close()

def plot_arrows_list_threshold(points_list, offset_list, joint_list, title_name='default', dpi=200, s=5, lw=5, length=0.5, threshold=0.2):
    """
    points: [N, 3]
    offset: [N, 3]
    joint : [P0, ll], a list, array

    """
    fig     = plt.figure(dpi=dpi)
    cmap    = plt.cm.jet
    colors  = cmap(np.linspace(0., 1., 5))
    c_set = ['r', 'g', 'b', 'k', 'm']
    all_poss=['.','o','v','^','>','<','s','p','*','h','H','D','d','1','','']
    ax = plt.subplot(1, 1, 1, projection='3d')
    for i in range(len(points_list)):
        points = points_list[i]
        p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[n], s=s, c='c')
        if joint_list[i] is not []:
            for m in range(len(joint_list[i])):
                offset = offset_list[i][m]
                joint  = joint_list[i][m]
                offset_norm = np.linalg.norm(offset, axis=1)
                idx = np.where(offset_norm<threshold)[0]
                ax.quiver(points[idx, 0], points[idx, 1], points[idx, 2], offset[idx, 0], offset[idx, 1], offset[idx, 2], color=c_set[i])
                ax.plot3D([joint[0][0, 0]- length * joint[1][0], joint[0][0, 0] + length * joint[1][0]], \
                          [joint[0][0, 1]- length * joint[1][1], joint[0][0, 1] + length * joint[1][1]], \
                          [joint[0][0, 2]- length * joint[1][2], joint[0][0, 2] + length * joint[1][2]],  linewidth=lw, c='blue')
    # set_axes_equal(ax
    plt.title(title_name)
    plt.show()
    plt.close()


def hist_show(values, labels, tick_label, axes_label, title_name, total_width=0.5, dpi=300, save_fig=False, sub_name='seen'):
    x     = list(range(len(values[0])))
    n     = len(labels)
    width = total_width / n
    colors=['r', 'b', 'g', 'k', 'y']
    fig = plt.figure(figsize=(20, 5), dpi=dpi)
    ax = plt.subplot(111)

    for i, num_list in enumerate(values):
        if i == int(n/2):
            plt.xticks(x, tick_label, rotation='vertical', fontsize=5)
        plt.bar(x, num_list, width=width, label=labels[i], fc=colors[i])
        if len(x) < 10:
            for j in range(len(x)):
                if num_list[j] < 0.30:
                    ax.text(x[j], num_list[j], '{0:0.04f}'.format(num_list[j]), color='black', fontsize=2)
                else:
                    ax.text(x[j], 0.28, '{0:0.04f}'.format(num_list[j]), color='black', fontsize=2)
        for j in range(len(x)):
            x[j] = x[j] + width
    if title_name.split()[0] == 'rotation':
        ax.set_ylim(0, 30)
    elif title_name.split()[0] == 'translation':
        ax.set_ylim(0, 0.10)
    elif title_name.split()[0] == 'ADD':
        ax.set_ylim(0, 0.10)
    plt.title(title_name)
    plt.xlabel(axes_label[0], fontsize=8, labelpad=0)
    plt.ylabel(axes_label[1], fontsize=8, labelpad=5)
    plt.legend()
    plt.show()
    if save_fig:
        if not os.path.exists('./results/test/'):
            os.makedirs('./results/test/')
        fig.savefig('./results/test/{}_{}.png'.format(title_name, sub_name), pad_inches=0)
    plt.close()


def draw(img, imgpts, axes=None, color=None):
    imgpts = np.int32(imgpts).reshape(-1, 2)


    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([1, 3, 7, 5],[3, 7, 5, 1]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 2)


    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip([0, 2, 6, 4],[1, 3, 7, 5]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 2)


    # finally, draw top layer in color
    for i, j in zip([0, 2, 6, 4],[2, 6, 4, 0]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 2)

    # draw axes
    if axes is not None:
        img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
        img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
        img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3) ## y last

    return img

def draw_text(draw_image, bbox, text, draw_box=False):
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    thickness = 1

    retval, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)

    bbox_margin = 10
    text_margin = 10

    text_box_pos_tl = (min(bbox[1] + bbox_margin, 635 - retval[0] - 2* text_margin) , min(bbox[2] + bbox_margin, 475 - retval[1] - 2* text_margin))
    text_box_pos_br = (text_box_pos_tl[0] + retval[0] + 2* text_margin,  text_box_pos_tl[1] + retval[1] + 2* text_margin)

    # text_pose is the bottom-left corner of the text
    text_pos = (text_box_pos_tl[0] + text_margin, text_box_pos_br[1] - text_margin - 3)

    if draw_box:
        cv2.rectangle(draw_image,
                      (bbox[1], bbox[0]),
                      (bbox[3], bbox[2]),
                      (255, 0, 0), 2)

    cv2.rectangle(draw_image,
                  text_box_pos_tl,
                  text_box_pos_br,
                  (255,0,0), -1)

    cv2.rectangle(draw_image,
                  text_box_pos_tl,
                  text_box_pos_br,
                  (0,0,0), 1)

    cv2.putText(draw_image, text, text_pos,
                fontFace, fontScale, (255,255,255), thickness)

    return draw_image

def plot_distribution(d, labelx='Value', labely='Frequency', title_name='Mine', dpi=200, xlimit=None, put_text=False):
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

def viz_err_distri(val_gt, val_pred, title_name):
    if val_gt.shape[1] > 1:
        err = np.linalg.norm(val_gt - val_pred, axis=1)
    else:
        err = np.squeeze(val_gt) - np.squeeze(val_pred)
    plot_distribution(err, labelx='L2 error', labely='Frequency', title_name=title_name, dpi=160)


if __name__=='__main__':
    import numpy as np
    d = np.random.laplace(loc=15, scale=3, size=500)
    plot_distribution(d)
