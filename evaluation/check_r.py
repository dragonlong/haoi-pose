"""
.. _glyph_example:

Plotting Glyphs (Vectors or PolyData)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use vectors in a dataset to plot and orient glyphs/geometric objects.
"""

# sphinx_gallery_thumbnail_number = 4
import pyvista as pv
from pyvista import examples
import numpy as np
from os import makedirs, remove
from os.path import exists, join
###############################################################################
# Glyphying can be done via the :func:`pyvista.DataSetFilters.glyph` filter
def bp():
    import pdb;pdb.set_trace()
# mesh = examples.download_carotid().threshold(145, scalars="scalars")
#
# # Make a geometric object to use as the glyph
# geom = pv.Arrow()  # This could be any dataset
#
# # Perform the glyph
# glyphs = mesh.glyph(orient="vectors", scale="scalars", factor=0.005, geom=geom)
#
# # plot using the plotting class
# p = pv.Plotter()
# p.add_mesh(glyphs)
# # Set a cool camera position

# bp()
# p.show()

###############################################################################
# Another approach is to load the vectors directly to the mesh object and then
# access the :attr:`pyvista.DataSet.arrows` property.
# fn = '/home/dragon/Documents/ICML2021/results/preds/train_0_0002_0_0.txt'
fn = '/home/dragon/Documents/ICML2021/results/preds/0.59/train_2850_0002_850_0.txt'
point_normal_set = np.loadtxt(fn, delimiter=' ').astype(np.float32)
points = point_normal_set[:, :3]
r_mat  = point_normal_set[:, 4:].reshape(-1, 3, 3)
x_axis = np.matmul(np.array([[1.0, 0.0, 0.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
y_axis = np.matmul(np.array([[0.0, 1.0, 0.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))
z_axis = np.matmul(np.array([[0.0, 0.0, 1.0]])[np.newaxis, :, :], r_mat.transpose(0, 2, 1))

# # Create random XYZ points
# points = np.random.rand(100, 3)
# Make PolyData
point_cloud = pv.PolyData(points)
#
# def compute_vectors(mesh):
#     origin = mesh.center
#     vectors = mesh.points - origin
#     vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
#     return vectors
#
###############################################################################
bp()
point_cloud['vectors'] = x_axis[:, 0, :]
###############################################################################
# Now we can make arrows using those vectors using the glyph filter
# (see :ref:`glyph_example` for more details).
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
#
# # make cool swirly pattern
# vectors = np.vstack(
#     (
#         np.sin(sphere.points[:, 0]),
#         np.cos(sphere.points[:, 1]),
#         np.cos(sphere.points[:, 2]),
#     )
# ).T
#
# # add and scale
# sphere.vectors = vectors * 0.3
# p.add_mesh(sphere.arrows, scalars='GlyphScale', lighting=False, stitle="Vector Magnitude")
# p.add_mesh(sphere, color="grey", ambient=0.6, opacity=0.5, show_edges=False)
# # add and scale
# sphere.vectors = vectors * 0.5
# p.add_mesh(sphere.arrows, scalars='GlyphScale', lighting=False, stitle="Vector Magnitude1")
# # p.camera_position = [
# #     (84.58052237950857, 77.76332116787425, 27.208569926456548),
# #     (131.39486171068918, 99.871379394528, 20.082859824932008),
# #     (0.13483731007732908, 0.033663777790747404, 0.9902957385932576),
# # ]
p.show_grid()
p.show()
