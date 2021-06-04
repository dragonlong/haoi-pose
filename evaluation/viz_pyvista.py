import pyvista as pv
from pyvista import examples
import numpy as np
from os import makedirs, remove
from os.path import exists, join
from glob import glob
import scipy.io as sio
import argparse
# import matplotlib
# import matplotlib.pyplot as plt
###############################################################################
# Glyphying can be done via the :func:`pyvista.DataSetFilters.glyph` filter
def bp():
    import pdb;pdb.set_trace()
import __init__
from global_info import global_info
infos           = global_info()
my_dir          = infos.base_path
project_path    = infos.project_path
categories_id   = infos.categories_id
categories      = infos.categories
my_dir          = infos.base_path
delta_R         = infos.delta_R
delta_T         = infos.delta_T

"""
>>>>>>>>>>>>>>>>>>>>>>>  knowledge <<<<<<<<<<<<<<<<<<<<<<<
# 1. Make a geometric object to use as the glyph
geom = pv.Arrow()  # This could be any dataset

# # Perform the glyph
# 2. glyphs = mesh.glyph(orient="vectors", scale="scalars", factor=0.005, geom=geom)
#
# plot using the plotting class
p = pv.Plotter()
p.add_mesh(glyphs)
# Set a cool camera position

# 3. sphere
sphere = pv.Sphere(radius=3.14)

# make cool swirly pattern
vectors = np.vstack(
    (
        np.sin(sphere.points[:, 0]),
        np.cos(sphere.points[:, 1]),
        np.cos(sphere.points[:, 2]),
    )
).T

# 4. add and scale
sphere.vectors = vectors * 0.3
p.add_mesh(sphere.arrows, scalars='GlyphScale', lighting=False, stitle="Vector Magnitude")
p.add_mesh(sphere, color="grey", ambient=0.6, opacity=0.5, show_edges=False)
# add and scale
sphere.vectors = vectors * 0.5
p.add_mesh(sphere.arrows, scalars='GlyphScale', lighting=False, stitle="Vector Magnitude1")
p.camera_position = [
    (84.58052237950857, 77.76332116787425, 27.208569926456548),
    (131.39486171068918, 99.871379394528, 20.082859824932008),
    (0.13483731007732908, 0.033663777790747404, 0.9902957385932576),
]

# 5. plot arrows
# Now we can make arrows using those vectors using the glyph filter
# (see :ref:`glyph_example` for more details).
cent = np.random.random((100, 3))
direction = np.random.random((100, 3))
pyvista.plot_arrows(cent, direction)

# 6. multiple windows
p = pv.Plotter(shape=(1,2))
p.add_mesh(mesh, scalars='Spatial Point Data', show_edges=True)
p.subplot(0,1)
p.add_mesh(mesh, scalars='Spatial Cell Data', show_edges=True)
p.show(screenshot='point_vs_cell_data.png')

# 7. save gif
p.show(auto_close=False)
path = p.generate_orbital_path(n_points=36, shift=mesh.length)
p.open_gif("orbit.gif")
p.orbit_on_path(path, write_frames=True)

# path = p.generate_orbital_path(factor=2.0, n_points=36, viewup=viewup, shift=0.2)
# p.open_gif("orbit.gif")
# p.orbit_on_path(path, write_frames=True, viewup=viewup)
"""
# def get_arrows():
    # p.add_mesh(arrows, color='blue')
    # point_cloud['vectors'] = y_axis[:, 0, :]
    # arrows1 = point_cloud.glyph(orient='vectors', scale=False, factor=0.15,)
    # p.add_mesh(arrows1, color='red')
    # p.add_point_labels([point_cloud.center,], ['Center',], point_color='yellow', point_size=20, font_size=16)

def get_axis(r_mat):
    x_axis = np.matmul(np.array([[1.0, 0.0, 0.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
    y_axis = np.matmul(np.array([[0.0, 1.0, 0.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
    z_axis = np.matmul(np.array([[0.0, 0.0, 1.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
    return x_axis, y_axis, z_axis

class simple_config(object):
    def __init__(self, target_entry=None, exp_num=None, target_category=None, name_dset=None, icp_method_type=0):
        self.canonical_path=f'{my_dir}/data/modelnet40aligned/EvenAlignedModelNet40PC'
        if target_entry is None:
            self.exp_num    = exp_num
            self.target_category = target_category
            self.name_dset  = name_dset
            self.symmetry_type = 0
            # if name_dset == 'modelnet40aligned':
            #     self.canonical_path=f'{my_dir}/data/modelnet40aligned/EvenAlignedModelNet40PC'
            # elif name_dset == 'modelnet40new':
            #     self.canonical_path=f'{my_dir}/data/modelnet40new/render/{target_category}/test/gt'
        else:
            target_category=target_entry.split('_')[1]
            if target_entry.split('_')[0] == 'complete':
                name_dset='modelnet40aligned'
            else:
                name_dset='modelnet40new'
            if name_dset == 'modelnet40aligned':
                # self.canonical_path=f'{my_dir}/data/modelnet40aligned/EvenAlignedModelNet40PC'
                if self.target_category == 'airplane':
                    self.exp_num    = '0.813'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                elif self.target_category == 'car':
                    self.exp_num    = '0.851'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                elif self.target_category == 'chair':
                    self.exp_num    = '0.8581'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                elif self.target_category == 'sofa':
                    self.exp_num    = '0.8591'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                elif self.target_category == 'bottle':
                    self.exp_num    = '0.8562'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                    self.symmetry_type = 1
                    self.chosen_axis = 'z'
            elif name_dset == 'modelnet40new':
                # self.canonical_path=f'{my_dir}/data/modelnet40new/render/{target_category}/test/gt'
                if self.target_category == 'airplane':
                    self.exp_num    = '0.913r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                elif self.target_category == 'car':
                    self.exp_num    = '0.921r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                elif self.target_category == 'chair':
                    self.exp_num    = '0.951r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                elif self.target_category == 'sofa':
                    self.exp_num    = '0.961r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                elif self.target_category == 'bottle':
                    self.exp_num    = '0.941r'     # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
                    self.symmetry_type = 1
                    self.chosen_axis = 'z'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_entry', default='complete_airplane', help='yyds')
    args = parser.parse_args()
    np.random.seed(2)
    pv.set_plot_theme("document")
    off_screen = False
    color      = 'gold' #'deepskyblue'
    target_entry    = args.target_entry # 'complete_airplane' # '0.961r' #'0.94r' # 0.92 #0.81 # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
    k = 1.3
    k1 = 1.05
    point_size   = 10
    font_size    = 18
    query_keys   = ['canon', 'target', 'input', 'pred'] #
    fpath   = f'/home/dragon/Documents/ICML2021/results/preds/{target_entry}/generation/' # generation/generation
    fnames = {}
    for key in query_keys:
        fnames[key] = sorted(glob(f'{fpath}/*_0*_{key}*txt'))

    idxs = np.random.permutation(len(fnames['canon']))
    # p = pv.Plotter(off_screen=off_screen, lighting='light_kit', shape=(1, 1))
    num  = 10
    colors = np.random.rand(num, 3)
    colors_pred = np.random.rand(num, 3)
    p = pv.Plotter(notebook=0, shape=(2, 10), border=False)
    if '0.8' in args.target_entry:
        cfg = simple_config(exp_num=args.target_entry, target_category='airplane', name_dset='modelnet40aligned')
    else:
        cfg = simple_config(exp_num=args.target_entry, target_category='airplane', name_dset='modelnet40new')
    # cfg = simple_config(target_entry)
    # get delta_r, delta_t
    entry_key = f'{cfg.exp_num}_{cfg.name_dset}_{cfg.target_category}'
    print(entry_key)
    delta_r = np.squeeze(delta_R[entry_key]) # I --> delta_r,
    if entry_key in delta_T:
        delta_t = delta_T[entry_key]
    else:
        delta_t = None
    for i in range(num):
        p.subplot(0,i)
        index = idxs[i]
        fn = fnames['canon'][index]
        if delta_t is not None:
            point_cloud = pv.PolyData((np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3] - 0.5 - delta_t.reshape(1, 3) ) @ delta_r.T + 0.5)
        else:
            point_cloud = pv.PolyData((np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3] - 0.5) @ delta_r.T + 0.5)
        p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)
        p.subplot(1,i)
        instance_id = fn.split('.txt')[0].split('/')[-1].split('_')[2]
        complete_fname = cfg.canonical_path + f'/{cfg.target_category}/test/{cfg.target_category}_{instance_id}.mat'

        pc = sio.loadmat(complete_fname)['pc']
        boundary_pts = [np.min(pc, axis=0), np.max(pc, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        pc_canon = (pc - center_pt.reshape(1, 3))/length_bb
        pc_canon = pc_canon + 0.5 # NOCS space
        pc_canon = np.random.permutation(pc_canon)[:3072]
        point_cloud = pv.PolyData(pc_canon.astype(np.float32)[:, :3])
        # complete_fname = glob(fpath.replace('partial', 'complete') + f'*{instance_id}_target*')
        # point_cloud = pv.PolyData(np.loadtxt(complete_fname[0], delimiter=' ').astype(np.float32)[:, :3])
        p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)

        # p.subplot(2,i)
        # fn = fnames['canon'][index].replace('canon', 'input')
        # point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
        # p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)
        #
        # p.subplot(3,i)
        # p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)
        # fn = fnames['canon'][index].replace('canon', 'pred')
        # point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
        # p.add_mesh(point_cloud, color=colors_pred[i], point_size=point_size, render_points_as_spheres=True)

    p.link_views()  # link all the views
    p.camera_position = [(0.5, 0.5, 5), (0.5, 0.5, 0.5), (-1, 0, 0)]

    p.show(auto_close=False)
    print(p.camera_position)
    p.open_gif(f"./viz/{entry_key}.gif")

    nframe = 1
    for i in range(nframe):
        p.camera_position = [(0.5, 0.5, 5), (0.5, 0.5, 0.5), (-1, 0, 0)]
        p.write_frame()

    p.close()
    if cfg.name_dset == 'modelnet40new':
        p = pv.Plotter(notebook=0, shape=(2, 10), border=False)

        cfg = simple_config(target_entry)
        # get delta_r, delta_t
        entry_key = f'{cfg.exp_num}_{cfg.name_dset}_{cfg.target_category}'
        delta_r = np.squeeze(delta_R[entry_key]) # I --> delta_r,
        if entry_key in delta_T:
            delta_t = delta_T[entry_key]
        else:
            delta_t = None
        for i in range(num):
            # p.subplot(0,i)
            index = idxs[i]
            # fn = fnames['canon'][index]
            # if delta_t is not None:
            #     point_cloud = pv.PolyData((np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3] - 0.5 - delta_t.reshape(1, 3) ) @ delta_r.T + 0.5)
            # else:
            #     point_cloud = pv.PolyData((np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3] - 0.5) @ delta_r.T + 0.5)
            # p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)
            # p.subplot(1,i)
            # instance_id = fn.split('.txt')[0].split('/')[-1].split('_')[2]
            # complete_fname = cfg.canonical_path + f'/{cfg.target_category}/test/{cfg.target_category}_{instance_id}.mat'
            #
            # pc = sio.loadmat(complete_fname)['pc']
            # boundary_pts = [np.min(pc, axis=0), np.max(pc, axis=0)]
            # center_pt = (boundary_pts[0] + boundary_pts[1])/2
            # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
            # pc_canon = (pc - center_pt.reshape(1, 3))/length_bb
            # pc_canon = pc_canon + 0.5 # NOCS space
            # pc_canon = np.random.permutation(pc_canon)[:3072]
            # point_cloud = pv.PolyData(pc_canon.astype(np.float32)[:, :3])
            # # complete_fname = glob(fpath.replace('partial', 'complete') + f'*{instance_id}_target*')
            # # point_cloud = pv.PolyData(np.loadtxt(complete_fname[0], delimiter=' ').astype(np.float32)[:, :3])
            # p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)

            p.subplot(0,i)
            fn = fnames['canon'][index].replace('canon', 'input')
            point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
            p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)

            p.subplot(1,i)
            p.add_mesh(point_cloud, color=colors[i], point_size=point_size, render_points_as_spheres=True)
            fn = fnames['canon'][index].replace('canon', 'pred')
            point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
            p.add_mesh(point_cloud, color=colors_pred[i], point_size=point_size, render_points_as_spheres=True)

        p.link_views()  # link all the views
        p.camera_position = [(0.5, 0.5, 5), (0.5, 0.5, 0.5), (-1, 0, 0)]

        p.show(auto_close=False)
        print(p.camera_position)
        p.open_gif(f"./viz/{entry_key}.gif")

        nframe = 1
        for i in range(nframe):
            p.camera_position = [(0.5, 0.5, 5), (0.5, 0.5, 0.5), (-1, 0, 0)]
            p.write_frame()

        p.close()
    # p = pv.Plotter(off_screen=off_screen, lighting='light_kit')
    # for i in idxs:
    #     fn = fnames['canon'][i].replace('canon', 'target')
    #     point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3]+0.4*i)
    #     p.add_mesh(point_cloud, color=np.random.rand(3), point_size=5, render_points_as_spheres=True)
    # p.add_title('human-aligned shape space(100 unseen)', font_size=font_size)
    # p.show()

    # for fn in fnames['canon']:
    #     points = {}
    #     for key in query_keys:
    #         point_normal_set = np.loadtxt(fn.replace('canon', key), delimiter=' ').astype(np.float32)
    #         pts = point_normal_set[:, :3]
    #         # if key == 'input':
    #         #     refer_shift = pts.mean(axis=0, keepdims=True)
    #         # if key == 'pred':
    #         #     pts = pts + refer_shift
    #         # r_mat  = point_normal_set[:, 4:].reshape(-1, 3, 3)
    #         points[key] = pv.PolyData(pts)
    #     # point_cloud['vectors'] = x_axis[:, 0, :]
    #     # arrows = point_cloud.glyph(orient='vectors', scale=False, factor=0.15,)
    #     p = pv.Plotter(off_screen=off_screen, lighting='light_kit', shape=window_shape)
    #     p.add_mesh(points['target'], color='r', point_size=15, render_points_as_spheres=True)
    #     # p.camera_position = [(0.1*k, 1.8*k, 2.0*k),
    #     #                  (0.5, 0.5, 0.5),
    #     #                  (-0.9814055156571295, -0.1437895877734097, -0.12715253943872534)]
    #     p.camera_position = [(0.42189609206584944, 0.3720949834155453, 3.312479348599398),
    #                      (0.5, 0.5, 0.5),
    #                      (-0.999559884631558, 0.011789591923156638, -0.027222096863255326)]
    #     # p.add_legend([['nocs', 'r']], bcolor=(1.0, 1.0, 1.0))
    #     p.add_title('input(canonicalized)', font_size=font_size)
    #     p.show_grid()
    #
    #     p.subplot(0,1)
    #     p.add_mesh(points['canon'], color='g', point_size=15, render_points_as_spheres=True)
    #     # p.camera_position = [(0.1*k, 1.8*k, 2.0*k),
    #     #                  (0.5, 0.5, 0.5),
    #     #                  (-0.9814055156571295, -0.1437895877734097, -0.12715253943872534)]
    #     p.camera_position = [(1.6224649540075546*k1, 2.558959462540017*k1, -0.18487386521674765*k1),
    #                      (0.5058576315641403, 0.5140270888805389, 0.5149073377251625),
    #                      (-0.1485912819656584, -0.24657874854598089, -0.9576635900405216)]
    #
    #     # p.add_legend([['pred', 'g']], bcolor=(1.0, 1.0, 1.0))
    #     p.add_title('predicted shape', font_size=font_size)
    #     p.show_grid()
    #     p.subplot(0,2)
    #     sphere = pv.Sphere(radius=0.1)
    #     p.add_mesh(sphere,  color='b')
    #     p.add_mesh(points['pred'],  color='g', point_size=15, render_points_as_spheres=True)
    #     p.add_mesh(points['input'], color='r', point_size=15, render_points_as_spheres=True)
    #     p.add_legend([['pred', 'g'], ['input', 'r']], bcolor=(1.0, 1.0, 1.0))
    #     p.add_title('pose estimation', font_size=font_size)
    #     p.show_grid()
    #
    #     # p.subplot(0,3)
    #     # sphere = pv.Sphere(radius=0.1)
    #     # p.add_mesh(sphere,  color='b')
    #     #
    #     # p.add_mesh(points['icp'],  color='g', point_size=15, render_points_as_spheres=True)
    #     # p.add_mesh(points['input'], color='r', point_size=15, render_points_as_spheres=True)
    #     # p.add_legend([['icp', 'g'], ['input', 'r']], bcolor=(1.0, 1.0, 1.0))
    #     # p.add_title('icp estimation', font_size=font_size)
    #     # p.show_grid()
    #     cpos = p.show(screenshot='test.png', window_size=(1980, 920))
    #     print(cpos)
