
import _init_paths
from common.vis_utils import plot3d_pts, plot2d_img, plot_arrows, plot_arrows_list, plot_hand_w_object
from common.data_utils import get_demo_h5, get_full_test, save_objmesh, fast_load_obj, get_obj_mesh

import matplotlib.pyplot as plt

obj= fast_load_obj(open('/home/dragon/Documents/ICML2021/data/hands/canonical_hand.obj', 'rb'))[0] # why it is [0]
obj_verts = obj['vertices']
obj_faces = obj['faces']
plot_hand_w_object(obj_verts/2, obj_faces, hand_verts=obj_verts/2, hand_faces=obj_faces, s=5**2, mode='continuous', save=False)
plt.show()
