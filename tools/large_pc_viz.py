#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import yaml
import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.geometry import create_box
from vispy.visuals.transforms import MatrixTransform, STTransform

import numpy as np
import time
from matplotlib import pyplot as plt
def breakpoint():
    import pdb; pdb.set_trace()

DATA = yaml.safe_load(open('./config/semantic-kitti-all.yaml', 'r'))


class PointVis:
  """Class that creates and handles a visualizer for a pointcloud"""
  def __init__(self, target_pts=None, labels=None, viz_dict=None, viz_point=True, viz_label=True, viz_joint=False, viz_box=False, color_map=None):
    self.viz_point = viz_point
    self.viz_label = viz_label
    self.viz_joint = viz_joint
    self.viz_box   = viz_box
    self.viz_label  = viz_label
    #
    self.color_map=DATA["color_map"]
    self.learning_map=DATA["learning_map"]
    self.learning_map_inv=DATA["learning_map_inv"]

    self.reset(sem_color_dict=self.color_map)

    self.update_scan(target_pts, labels, viz_dict)

  def reset(self, sem_color_dict=None):
    """ Reset. """
    # new canvas prepared for visualizing data
    self.map_color(sem_color_dict=sem_color_dict)
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_view.camera = 'turntable'

    self.scan_vis = visuals.Markers()
    self.scan_view.add(self.scan_vis)

    if self.viz_joint:
      self.joint_vis = visuals.Arrow(connect='segments', arrow_size=18, color='blue', width=10, arrow_type='angle_60')
      self.arrow_length = 10
      self.scan_view.add(self.joint_vis)

    if self.viz_box:
      vertices, faces, outline = create_box(width=1, height=1, depth=1, width_segments=1, height_segments=1, depth_segments=1)
      vertices['color'][:, 3]=0.2
      # breakpoint()
      self.box = visuals.Box(vertex_colors=vertices['color'],
                                   edge_color='b')
      self.box.transform = STTransform(translate=[-2.5, 0, 0])
      self.theta = 0
      self.phi   = 0
      self.scan_view.add(self.box)
    visuals.XYZAxis(parent=self.scan_view.scene)

    # add nocs
    if self.viz_label:
      print("Using nocs in visualizer")
      self.nocs_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.nocs_view, 0, 1)
      self.label_vis = visuals.Markers()
      self.nocs_view.camera = 'turntable'
      self.nocs_view.add(self.label_vis)
      visuals.XYZAxis(parent=self.nocs_view.scene)
      self.nocs_view.camera.link(self.scan_view.camera)

  def update_scan(self, points, labels, viz_dict=None):
    # then change names
    self.canvas.title = "scan "
    if viz_dict is not None:
        lengths    = viz_dict['input_stack_lengths'][0]
        target_pts = viz_dict['input_points'][0]
        gt_labels  = viz_dict['label'].astype(np.int32)
        gt_labels  = self.map(gt_labels, self.learning_map_inv)
        sum=0
        new_lengths = []
        for x in lengths:
          sum = x+sum
          new_lengths.append(sum)
        target_pts_list = np.split(target_pts, new_lengths, axis=0)
        gt_labels_list  = np.split(gt_labels, new_lengths, axis=0)
        offsets = np.max(target_pts_list[0], axis=0).reshape(1, 3)
        pts1    = target_pts_list[0] - offsets
        labels1 = gt_labels_list[0]
        pts2     = np.concatenate(target_pts_list[0:2], axis=0) - offsets
        labels2  = np.concatenate(gt_labels_list[0:2], axis=0)

    else:
        pts1 = points
        pts2 = points
        labels1 = labels
        labels2 = labels
    label_colors = self.sem_color_lut[labels1]
    label_colors = label_colors.reshape((-1, 3))
    self.scan_vis.set_data(pts1,
                      face_color=label_colors[..., ::-1],
                      edge_color=label_colors[..., ::-1],
                      size=5)

    # plot nocs
    if self.viz_label:
        label_colors = self.sem_color_lut[labels2]
        label_colors = label_colors.reshape((-1, 3))
        self.label_vis.set_data(pts2,
                          face_color=label_colors[..., ::-1],
                          edge_color=label_colors[..., ::-1],
                          size=5)

    if self.viz_joint:
      self.update_joints()

    if self.viz_box:
      self.update_boxes()

  def map_color(self, max_classes=30, sem_color_dict=None):
    # make semantic colors
    if sem_color_dict:
      max_sem_key = 0
      for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
          max_sem_key = key + 1
      self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
      for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
    else:
      # otherwise make random
      max_sem_key = max_classes
      self.sem_color_lut = np.random.uniform(low=0.0,
                                             high=1.0,
                                             size=(max_sem_key, 3))
      # force zero to a gray-ish color
      self.sem_color_lut[0] = np.full((3), 0.2)
      self.sem_color_lut[4] = np.full((3), 0.6)
      self.sem_color_lut[1] = np.array([1.0, 0.0, 0.0])
      self.sem_color_lut[2] = np.array([0.0, 0.0, 1.0])

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_joints(self, joints=None):
    # plot
    if joints is not None:
      start_coords  = joints['p'].reshape(1, 3)
      point_towards = start_coords + joints['l'].reshape(1, 3)
    else:
      start_coords  = np.array([[1, 0, 0],
                                [-1, 0, 0]])
      point_towards = np.array([[0, 0, 1], [0, 0, 1]])
    direction_vectors = (start_coords - point_towards).astype(
        np.float32)
    norms = np.sqrt(np.sum(direction_vectors**2, axis=-1))
    direction_vectors[:, 0] /= norms
    direction_vectors[:, 1] /= norms
    direction_vectors[:, 2] /= norms

    vertices = np.repeat(start_coords, 2, axis=0)
    vertices[::2]  = vertices[::2] + ((0.5 * self.arrow_length) * direction_vectors)
    vertices[1::2] = vertices[1::2] - ((0.5 * self.arrow_length) * direction_vectors)

    self.joint_vis.set_data(
        pos=vertices,
        arrows=vertices.reshape((len(vertices)//2, 6)),
    )

  def update_boxes(self):
    pass

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()

  def map(self, label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

if __name__ == '__main__':
    N = 100000
    target_pts = (np.random.rand(N,3) - 0.5) * 50
    labels = np.random.rand(N) * 20          # labels are encoded as color
    labels = labels.astype(np.int8)
    vis = PointVis(target_pts=target_pts, labels=labels, viz_joint=False, viz_box=False)
    # vis.run()
    input("Press Enter to continue...")

    # #>>>>>>>>>>>>> in case we need to save the visualizations
    # img=self.canvas.render()
    # directory = f'/home/dragon/Dropbox/cvpr2021/viz/kpconv/pictures/{file_name}'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # s_r = 2**k * 0.06 * 2.5
    # io.write_png(f"{directory}/patch{i}_scale{k}_point{j}_r={s_r}.png",img) # patch, scale, points j
    # print('saving to ', f"{directory}/patch{i}_scale{k}_point{j}_r={s_r}.png")
    print('checking data')
