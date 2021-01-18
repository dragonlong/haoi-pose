import numpy as np
import _init_paths
from common.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list, plot_hand_w_object


outfile = '/home/dragon/Documents/ICML2021/data/external/output/ShapeNet.build/03001627/4_points/1a6f615e8b1b5ae4dbbc9440457e303e.npz'
npzfile = np.load(outfile)
# print(npzfile.points)
print(np.unpackbits(npzfile['occupancies']).shape)
print(npzfile['points'].shape)
plot3d_pts([[npzfile['points']]], [['points']], color_channel=[[  np.concatenate([np.unpackbits(npzfile['occupancies'])[:, np.newaxis]]*3, axis=1)  ]])
