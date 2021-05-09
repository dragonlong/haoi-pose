import numpy as np
import pyrender
import trimesh
import meshplot

import __init__
from global_info import global_info

infos     = global_info()
my_dir    = infos.base_path
second_path = infos.second_path

def viz_mesh(vertices=None, faces=None, pts=None, title_name='default', labels=None):
    scene = pyrender.Scene()
    if vertices is not None:
        finale_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_vis = pyrender.Mesh.from_trimesh(finale_mesh)
        scene.add(mesh_vis)

    if pts is not None:
        if pts.shape[0] > 5000:
            if labels is not None:
                colors = np.zeros(pts.shape)
                colors[labels < 0.1, 2] = 1
                colors[labels > 0.1, 1] = 1
                cloud = pyrender.Mesh.from_points(pts, colors=colors)
            else:
                cloud = pyrender.Mesh.from_points(pts)
            scene.add(cloud)
        else:
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [1.0, 1.0, 0.0]
            tfs = np.tile(np.eye(4), (len(pts), 1, 1))
            tfs[:,:3,3] = pts
            m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(m)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5, show_world_axis=False, window_title=title_name)

if __name__ == '__main__':
    template_pth = '../1024_spheres.npy'
    mesh_dict = np.load(template_pth, allow_pickle=True).item()
    vertices = mesh_dict['vertices']
    faces    = mesh_dict['faces']
    # viz_mesh(vertices, faces)
    preds_dir = second_path + f'/results/preds/0.8161/'
    in1 = f'{preds_dir}/test_31000_21_input.txt'
    out1= f'{preds_dir}/test_31000_21_pred.txt'
    pts = np.loadtxt(in1)[:, :3]
    new_vertices = np.loadtxt(out1)[:, :3]
    print(pts.shape, new_vertices.shape)
    viz_mesh(pts=pts, title_name='input')
    viz_mesh(pts=new_vertices, title_name='transformed pred')
    viz_mesh(new_vertices, faces)
    viz_mesh(new_vertices, faces, pts)
