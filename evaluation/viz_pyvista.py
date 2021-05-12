import pyvista as pv
from pyvista import examples
import numpy as np
from os import makedirs, remove
from os.path import exists, join
from glob import glob
# import matplotlib
# import matplotlib.pyplot as plt
###############################################################################
# Glyphying can be done via the :func:`pyvista.DataSetFilters.glyph` filter
def bp():
    import pdb;pdb.set_trace()

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

if __name__ == '__main__':
    pv.set_plot_theme("document")
    off_screen = False
    color      = 'gold' #'deepskyblue'
    exp_num    = 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
    window_shape = (1,3)
    k = 1.3
    k1 = 1.05
    font_size    = 18
    query_keys   = ['canon', 'target', 'input', 'pred']
    fpath   = f'/home/dragon/Documents/ICML2021/results/preds/{exp_num}/generation/' # generation/generation
    ###############################################################################
    # Another approach is to load the vectors directly to the mesh object and then
    # access the :attr:`pyvista.DataSet.arrows` property.
    # fn = '/home/dragon/Documents/ICML2021/results/preds/train_0_0002_0_0.txt'
    # fn = '/home/dragon/Documents/ICML2021/results/preds/0.59/train_2850_0002_850_0.txt'
    # fn = '/home/dragon/Documents/ICML2021/results/preds/0.63/train_990_0002_0_0.txt'
    # fn = '/home/dragon/Documents/ICML2021/results/preds/0.61/train_1000_0002_0_0.txt'
    fnames = {}
    for key in query_keys:
        fnames[key] = sorted(glob(f'{fpath}/*{key}*txt'))
    #
    # idxs = np.random.permutation(len(fnames['canon']))
    # p = pv.Plotter(off_screen=off_screen, lighting='light_kit')
    # for i in idxs:
    #     fn = fnames['canon'][i]
    #     point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
    #     p.add_mesh(point_cloud, color=np.random.rand(3), point_size=5, render_points_as_spheres=True)
    # p.add_title('predicted shape space(100 unseen)', font_size=font_size)
    # p.show_grid()
    # p.show()
    #
    # p = pv.Plotter(off_screen=off_screen, lighting='light_kit')
    # for i in idxs:
    #     fn = fnames['canon'][i].replace('canon', 'target')
    #     point_cloud = pv.PolyData(np.loadtxt(fn, delimiter=' ').astype(np.float32)[:, :3])
    #     p.add_mesh(point_cloud, color=np.random.rand(3), point_size=5, render_points_as_spheres=True)
    # p.add_title('human-aligned shape space(100 unseen)', font_size=font_size)
    # p.show_grid()
    # p.show()
    for fn in fnames['canon']:
        points = {}
        for key in query_keys:
            point_normal_set = np.loadtxt(fn.replace('canon', key), delimiter=' ').astype(np.float32)
            pts = point_normal_set[:, :3]
            if key == 'input':
                refer_shift = pts.mean(axis=0, keepdims=True)
            if key == 'pred':
                pts = pts + refer_shift
            # r_mat  = point_normal_set[:, 4:].reshape(-1, 3, 3)
            points[key] = pv.PolyData(pts)
        # point_cloud['vectors'] = x_axis[:, 0, :]
        # arrows = point_cloud.glyph(orient='vectors', scale=False, factor=0.15,)
        p = pv.Plotter(off_screen=off_screen, lighting='light_kit', shape=window_shape)
        p.add_mesh(points['target'], color='r', point_size=15, render_points_as_spheres=True)
        # p.camera_position = [(0.1*k, 1.8*k, 2.0*k),
        #                  (0.5, 0.5, 0.5),
        #                  (-0.9814055156571295, -0.1437895877734097, -0.12715253943872534)]
        p.camera_position = [(0.42189609206584944, 0.3720949834155453, 3.312479348599398),
                         (0.5, 0.5, 0.5),
                         (-0.999559884631558, 0.011789591923156638, -0.027222096863255326)]
        # p.add_legend([['nocs', 'r']], bcolor=(1.0, 1.0, 1.0))
        p.add_title('input(canonicalized)', font_size=font_size)
        p.show_grid()

        p.subplot(0,1)
        p.add_mesh(points['canon'], color='g', point_size=15, render_points_as_spheres=True)
        # p.camera_position = [(0.1*k, 1.8*k, 2.0*k),
        #                  (0.5, 0.5, 0.5),
        #                  (-0.9814055156571295, -0.1437895877734097, -0.12715253943872534)]
        p.camera_position = [(1.6224649540075546*k1, 2.558959462540017*k1, -0.18487386521674765*k1),
                         (0.5058576315641403, 0.5140270888805389, 0.5149073377251625),
                         (-0.1485912819656584, -0.24657874854598089, -0.9576635900405216)]

        # p.add_legend([['pred', 'g']], bcolor=(1.0, 1.0, 1.0))
        p.add_title('predicted shape', font_size=font_size)
        p.show_grid()
        p.subplot(0,2)
        sphere = pv.Sphere(radius=0.1)
        p.add_mesh(sphere,  color='b')
        p.add_mesh(points['pred'],  color='g', point_size=15, render_points_as_spheres=True)
        p.add_mesh(points['input'], color='r', point_size=15, render_points_as_spheres=True)
        p.add_legend([['pred', 'g'], ['input', 'r']], bcolor=(1.0, 1.0, 1.0))
        p.add_title('pose estimation', font_size=font_size)
        p.show_grid()
        cpos = p.show(screenshot='test.png', window_size=(1980, 920))
        print(cpos)
