import os
import random
import platform
import bpy
# import bpy_extras
from mathutils import Matrix, Vector
import math
import numpy as np
import scipy.io
import imageio
import glob
import xml.etree.ElementTree as ET

from math import pi ,sin, cos

# custom libs
import os.path as osp
import sys
import argparse

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def point_rotate_about_axis(pts, anchor, unitvec, theta):
    a, b, c = anchor.reshape(3)
    u, v, w = unitvec.reshape(3)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    ss =  u*x + v*y + w*z
    x_rotated = (a*(v**2 + w**2) - u*(b*v + c*w - ss)) * (1 - cos(theta)) + x * cos(theta) + (-c*v + b*w - w*y + v*z) * sin(theta)
    y_rotated = (b*(u**2 + w**2) - v*(a*u + c*w - ss)) * (1 - cos(theta)) + y * cos(theta) + (c*u - a*w + w*x - u*z) * sin(theta)
    z_rotated = (c*(u**2 + v**2) - w*(a*u + b*v - ss)) * (1 - cos(theta)) + z * cos(theta) + (-b*u + a*v - v*x + u*y) * sin(theta)
    rotated_pts = np.zeros_like(pts)
    rotated_pts[:, 0] = x_rotated
    rotated_pts[:, 1] = y_rotated
    rotated_pts[:, 2] = z_rotated

    return rotated_pts

def point_rotate_about_axis(pts, anchor, unitvec, theta):
    a, b, c = anchor.reshape(3)
    u, v, w = unitvec.reshape(3)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    ss =  u*x + v*y + w*z
    x_rotated = (a*(v**2 + w**2) - u*(b*v + c*w - ss)) * (1 - cos(theta)) + x * cos(theta) + (-c*v + b*w - w*y + v*z) * sin(theta)
    y_rotated = (b*(u**2 + w**2) - v*(a*u + c*w - ss)) * (1 - cos(theta)) + y * cos(theta) + (c*u - a*w + w*x - u*z) * sin(theta)
    z_rotated = (c*(u**2 + v**2) - w*(a*u + b*v - ss)) * (1 - cos(theta)) + z * cos(theta) + (-b*u + a*v - v*x + u*y) * sin(theta)
    rotated_pts = np.zeros_like(pts)
    rotated_pts[:, 0] = x_rotated
    rotated_pts[:, 1] = y_rotated
    rotated_pts[:, 2] = z_rotated

    return rotated_pts

class global_info(object):
    def __init__(self):
        self.name      = 'art6d'
        # self.datasets  = _DATASETS
        self.model_type= 'pointnet++'
        self.group_path= None
        self.name_dataset = 'shape2motion'

        # check dataset_name automactically
        group_path = None
        if platform.uname()[0] == 'Darwin':
            print("Now it knows it's in my local Mac")
            base_path = '/Users/DragonX/Downloads/ARC/6DPOSE'
        elif platform.uname()[1] == 'viz1':
            base_path = '/home/xiaolong/Downloads/6DPOSE'
        elif platform.uname()[1] == 'vllab3':
            base_path = '/mnt/data/lxiaol9/rbo'
        elif platform.uname()[1] == 'dragon':
            base_path = '/home/dragon/Documents/CVPR2020'
            second_path = '/home/dragon/Documents/ICML2021'
        else:
            base_path = '/work/cascades/lxiaol9/6DPOSE'
            group_path= '/groups/CESCA-CV'
            second_path = '/groups/CESCA-CV/ICML2021'

        self.render_path = second_path + '/data/render'
        self.viz_path  = second_path + '/data/images'
        self.hand_mesh = second_path + '/data/hands'
        self.hand_urdf = second_path + '/data/urdfs'
        self.hand_path = self.hand_mesh
        self.urdf_path = self.hand_urdf
        self.grasps_meta = second_path + '/data/grasps'
        self.mano_path   = '/home/dragon/Downloads/ICML2021/YCB_Affordance/data/mano'

        self.whole_obj = second_path + '/data/objs'
        self.part_obj  = base_path + '/dataset/{}/objects'.format(self.name_dataset)
        self.obj_urdf  = base_path + '/dataset/{}/urdf'.format(self.name_dataset)
        self.second_path = second_path
        self.base_path = base_path
        self.group_path= group_path
# from common.data_utils import get_model_pts, get_urdf
# >>>>>>>>>>>>>>> global env paths <<<<<<<<<<<<<<<<<<< #

infos     = global_info()
my_dir    = infos.base_path
second_path= infos.second_path
render_path = infos.render_path
viz_path  = infos.viz_path
hand_mesh = infos.hand_mesh
hand_urdf = infos.hand_urdf
grasps_meta  = infos.grasps_meta
mano_path    = infos.mano_path

whole_obj = infos.whole_obj
part_obj  = infos.part_obj
obj_urdf  = infos.obj_urdf

def breakpoint():
    import pdb;pdb.set_trace()


RENDERING_PATH = './'
MAX_CAMERA_DIST = 2
MAX_DEPTH = 1e8
FACTOR_DEPTH = 0.1

g_syn_light_num_lowbound = 4
g_syn_light_num_highbound = 6
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 12
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = 0
g_syn_light_elevation_degree_highbound = 90
g_syn_light_energy_mean = 3
g_syn_light_energy_std = 0.5
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 1

def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta):
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return (q1, q2, q3, q4)

def obj_centered_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

class BlenderRenderer(object):

    def __init__(self, viewport_size_x, viewport_size_y):
        '''
        viewport_size_x, viewport_size_y: rendering viewport resolution
        '''
        # remove the default cube
        bpy.ops.object.select_pattern(pattern="Cube")
        bpy.ops.object.delete()

        render_context = bpy.context.scene.render
        world  = bpy.context.scene.world
        camera = bpy.data.objects['Camera']

        # set the camera postion and orientation so that it is in
        # the front of the object
        camera.location = (1, 0, 0)

        # render setting
        render_context.resolution_percentage = 100
        world.horizon_color = (1, 1, 1)  # set background color to be white

        # set file name for storing temporary rendering result
        self.result_fn= '%s/render_result_%d.png' % (RENDERING_PATH, os.getpid())
        bpy.context.scene.render.filepath = self.result_fn

        # switch on nodes
        bpy.context.scene.use_nodes = True
        bpy.context.scene.view_settings.view_transform = 'Raw'
        tree = bpy.context.scene.node_tree
        links = tree.links

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')

        # gamma = tree.nodes.new('CompositorNodeGamma')
        # gamma.inputs[1].default_value = 2.2
        # links.new(rl.outputs[2], gamma.inputs[0])


        # create node viewer
        # v = tree.nodes.new('CompositorNodeViewer')
        # links.new(rl.outputs[2], v.inputs[0])  # link Image output to Viewer input

        # create map value layer node
        # map = tree.nodes.new(type="CompositorNodeMapValue")
        # map.size = [FACTOR_DEPTH]
        # map.use_min = True
        # map.min = [0]
        # map.use_max = True
        # map.max = [256]
        # links.new(rl.outputs[2], map.inputs[0])

        # create output node
        fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        fileOutput.base_path = "./new_data/0000"
        fileOutput.format.file_format = 'OPEN_EXR'
        fileOutput.format.color_depth= '32'
        fileOutput.file_slots[0].path = 'depth#'
        # links.new(map.outputs[0], fileOutput.inputs[0])
        links.new(rl.outputs[2], fileOutput.inputs[0])
        # links.new(gamma.outputs[0], fileOutput.inputs[0])

        self.render_context = render_context

        self.node_tree = tree
        self.fileOutput = fileOutput

        self.camera = camera
        self.model_loaded = False
        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y
        self.render_context.use_antialiasing = False

        self.dirname = 'new_data/0000'

    def _set_lighting(self, light_info=[], environment_energy=None):
        # clear default lights
        bpy.ops.object.select_by_type(type='LAMP')
        bpy.ops.object.delete(use_global=False)

        # set environment lighting
        bpy.context.scene.world.light_settings.use_environment_light = True
        bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

        # if light info is specified
        if len(light_info):
            bpy.context.scene.world.light_settings.environment_energy = environment_energy
            for info in light_info:
                light_azimuth_deg = info[0]
                light_elevation_deg = info[1]
                light_dist = info[2]
                light_energy = info[3]

                lx, ly, lz = obj_centered_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
                bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(lx, ly, lz))
                bpy.data.objects['Point'].data.energy = light_energy
        else: # randomly get a new set of lights
            bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(
                g_syn_light_environment_energy_lowbound, g_syn_light_environment_energy_highbound)

            # set point lights
            num_light = random.randint(g_syn_light_num_lowbound,g_syn_light_num_highbound)
            print(num_light)
            light_info = np.zeros((num_light, 4), dtype=np.float32)
            for i in range(num_light):
                light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
                light_elevation_deg  = np.random.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
                light_dist = np.random.uniform(g_syn_light_dist_lowbound, g_syn_light_dist_highbound)
                lx, ly, lz = obj_centered_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
                bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
                light_energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
                bpy.data.objects['Point'].data.energy = light_energy

                light_info[i, 0] = light_azimuth_deg
                light_info[i, 1] = light_elevation_deg
                light_info[i, 2] = light_dist
                light_info[i, 3] = light_energy

        self.environment_energy = bpy.context.scene.world.light_settings.environment_energy
        self.light_info = light_info


    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov, center=None):

        cx, cy, cz = obj_centered_camera_pos(distance_ratio * MAX_CAMERA_DIST, azimuth, altitude)
        q1 = camPosToQuaternion(cx, cy, cz)
        q2 = camRotQuaternion(cx, cy, cz, yaw)
        q = quaternionProduct(q2, q1)

        if center is not None:
            self.camera.location[0] = cx + center[0]
            self.camera.location[1] = cy + center[1]
            self.camera.location[2] = cz + center[2]
        else:
            self.camera.location[0] = cx
            self.camera.location[1] = cy
            self.camera.location[2] = cz

        self.camera.rotation_mode = 'QUATERNION'
        self.camera.rotation_quaternion[0] = q[0]
        self.camera.rotation_quaternion[1] = q[1]
        self.camera.rotation_quaternion[2] = q[2]
        self.camera.rotation_quaternion[3] = q[3]

        self.azimuth = azimuth
        self.elevation = altitude
        self.tilt = yaw
        self.distance = distance_ratio * MAX_CAMERA_DIST

    def setTransparency(self, transparency='SKY'):
        """ transparency is either 'SKY', 'TRANSPARENT'
        If set 'SKY', render background using sky color."""
        self.render_context.alpha_mode = transparency

    def makeMaterial(self, name, diffuse, specular, alpha):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color  = diffuse
        mat.diffuse_shader = 'LAMBERT'
        mat.diffuse_intensity = 1.0
        mat.specular_color = specular
        mat.specular_shader = 'COOKTORR'
        mat.specular_intensity = 0.5
        mat.alpha = alpha
        mat.ambient = 1
        mat.use_transparency = True
        mat.transparency_method = 'Z_TRANSPARENCY'
        mat.use_shadeless = True
        mat.use_face_texture = False
        return mat

    def selectModel(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_pattern(pattern="Camera")
        bpy.ops.object.select_all(action='INVERT')

    def printSelection(self):
        print(bpy.context.selected_objects)

    def clearModel(self):
        self.selectModel()
        bpy.ops.object.delete()

        # The meshes still present after delete
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        for item in bpy.data.materials:
            bpy.data.materials.remove(item)

    def loadModel(self, file_path):
        self.model_loaded = True
        try:
            if file_path.endswith('obj'):
                bpy.ops.import_scene.obj(filepath=file_path)
            elif file_path.endswith('3ds'):
                bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
            elif file_path.endswith('dae'):
                # Must install OpenCollada. Please read README.md
                bpy.ops.wm.collada_import(filepath=file_path)
            else:
                self.model_loaded = False
                raise Exception("Loading failed: %s" % (file_path))
        except Exception:
            self.model_loaded = False

    def loadModels(self, file_paths, scales, classes, filename=None, instance_info=None):
        self.model_loaded = True
        mesh = dict()
        num_parts = instance_info['num_parts']
        thetas   = instance_info['thetas']
        name_obj = instance_info['name_obj']
        instance = instance_info['instance']

        num = len(file_paths)
        height_max = -np.inf * np.ones((num,), dtype=np.float32)
        height_min = np.inf * np.ones((num,), dtype=np.float32)

        for i in range(num):
            file_path = file_paths[i]
            print('loading ', file_path)
            try:
                if file_path.endswith('obj'):
                    bpy.ops.import_scene.obj(filepath=file_path)
                elif file_path.endswith('3ds'):
                    bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
                elif file_path.endswith('dae'):
                    # Must install OpenCollada. Please read README.md for installation
                    bpy.ops.wm.collada_import(filepath=file_path)
                else:
                    # Other formats not supported yet
                    self.model_loaded = False
                    raise Exception("Loading failed: %s" % (file_path))

                # check limits
                for item in bpy.data.objects:
                    if item.type == 'MESH':
                        if item.name not in mesh:
                            mesh[item.name] = i
                            for vertex in item.data.vertices:
                                height_max[i] = max(height_max[i], scales[i] * vertex.co[1])
                                height_min[i] = min(height_min[i], scales[i] * vertex.co[1])

            except Exception:
                self.model_loaded = False
        # collect the vertices
        vertices = []
        for i in range(num):
            vertices.append(np.zeros((0, 3), dtype=np.float32))
        item_nocs=None
        for item in bpy.data.objects:
            if item.type == 'MESH':
                if mesh[item.name]==1:
                    item_nocs = item
                ind = mesh[item.name]
                for vertex in item.data.vertices:
                    vertices[ind] = np.append(vertices[ind], np.array(vertex.co).reshape((1,3)), axis = 0)

        # compute the boundary of objects
        Xlim = np.zeros((num, 2), dtype=np.float32)
        Ylim = np.zeros((num, 2), dtype=np.float32)
        Zlim = np.zeros((num, 2), dtype=np.float32)
        for i in range(0, num):
            print('---', i)
            Xlim[i, 0] = vertices[i][:, 0].min()
            Xlim[i, 1] = vertices[i][:, 0].max()
            Ylim[i, 0] = vertices[i][:, 1].min()
            Ylim[i, 1] = vertices[i][:, 1].max()
            Zlim[i, 0] = vertices[i][:, 2].min()
            Zlim[i, 1] = vertices[i][:, 2].max()
        XYZlim = np.stack([Xlim, Ylim, Zlim], axis=0) # 3, N, 2
        print('XYZlim has shape: ', XYZlim.shape)
        # breakpoint()
        XYZ_c = (XYZlim[:, :, 1] + XYZlim[:, :, 0])/2 # 3, N
        # breakpoint()
        XYZ_l = np.linalg.norm(XYZlim[:, :, 1] - XYZlim[:, :, 0], axis=0) # N
        self.mesh  = mesh # what use of this
        urdf_ins   = get_urdf("{}/{}/{}".format(obj_urdf, name_obj, instance))
        # >>>>>>>> get nocs material and transform the parts
        self.materials = {}
        # add new_material
        for item in bpy.data.objects:
            if item.type == 'MESH': # only collect material for nocs
                item.select = True
                item_color = item
                if mesh[item.name]==0:
                    item_color = item_nocs

                if len(item.data.vertex_colors)==0:
                    vcol_layer = item.data.vertex_colors.new()
                    for loop_index, loop in enumerate(item_color.data.loops):
                        loop_vert_index = loop.vertex_index
                        # color = Vector([0.5, 0.5, 0.5])
                        color = (item_color.data.vertices[loop_vert_index].co - Vector(XYZ_c[:, mesh[item_color.name]].tolist())) / XYZ_l[mesh[item_color.name]] + Vector([0.5, 0.5, 0.5])
                        vcol_layer.data[loop_index].color = color
                else:
                    vcol_layer = item.data.vertex_colors.active

                item.select = False
            pass
        # transform the objects
        translations = np.zeros((num, 3), dtype=np.float32)
        for item in bpy.data.objects:
            if item.type == 'MESH':
                k = int(classes[mesh[item.name]].split('_')[-1])
                if mesh[item.name] == 1:
                    item.select = True
                    bpy.ops.object.delete()

                if k > 99 or k==0:
                    continue
                # R = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
                anchor = np.array(urdf_ins['joint']['xyz'][k])
                univec = np.array(urdf_ins['joint']['axis'][k])
                theta  = thetas[k-1] * np.pi/2

                for vertex in item.data.vertices:
                    # rv = np.dot(R, np.array(vertex.co).reshape((3,1)))
                    rv = point_rotate_about_axis(np.array(vertex.co).reshape((1,3)), anchor, univec, theta)
                    rv = rv[0]
                    vertex.co[0] = rv[0]
                    vertex.co[1] = rv[1]
                    vertex.co[2] = rv[2]
                    # os.sys.exit(1)
        # # rescale the meshes
        # for item in bpy.data.objects:
        #     if item.type == 'MESH':
        #         ind = mesh[item.name]
        #         for vertex in item.data.vertices:
        #             vertex.co *= scales[ind]



        '''
        # rotate the objects and rescale the meshes
        for item in bpy.data.objects:
            bpy.context.scene.objects.active = item
            if item.type == 'MESH':
                ind = mesh[item.name]
                theta = thetas[ind]
                scale = scales[ind]

                bpy.ops.transform.rotate(value=theta, axis=(0, 0, 1))
                bpy.ops.transform.resize(value=(scale, scale, scale))
        '''

        # # add a transparent plane
        # V = np.zeros((0, 3), dtype=np.float32)
        # for item in bpy.data.objects:
        #     if item.type == 'MESH':
        #         for vertex in item.data.vertices:
        #             V = np.append(V, np.array(vertex.co).reshape((1,3)), axis = 0)

        # factor = 3
        # x1 = factor * np.min(V[:,0])
        # x2 = factor * np.max(V[:,0])
        # y1 = factor * np.min(V[:,2])
        # y2 = factor * np.max(V[:,2])
        # z = np.min(V[:,1])

        # verts = [(x1, y1, z), (x2, y1, z), (x2, y2, z), (x1, y2, z)]
        # faces = [(0, 1, 2, 3)]

        # mesh_data = bpy.data.meshes.new("cube_mesh_data")
        # obj = bpy.data.objects.new("plane", mesh_data)
        # bpy.context.scene.objects.link(obj)
        # bpy.context.scene.objects.active = obj
        # obj.select = True

        # mesh_data.from_pydata(verts, [], faces)
        # mesh_data.update()

        # mat = self.makeMaterial('transparent', (0.5,0.5,0.5), (0,0,0), 1)
        # obj.data.materials.append(mat)

        # save model
        self.obj_dict = mesh
        self.thetas = thetas
        # self.translations = translations
        self.scales = scales

        if filename:
            bpy.ops.export_scene.obj(filepath=filename, use_selection=False)

        return XYZlim


    # Build intrinsic camera parameters from Blender camera data
    def compute_intrinsic(self):
        '''
        w = self.render_context.resolution_x * self.render_context.resolution_percentage / 100.
        h = self.render_context.resolution_y * self.render_context.resolution_percentage / 100.
        K = Matrix().to_3x3()

        #a_u
        K[0][0] = w/2. / math.tan(self.camera.data.angle/2)
        ratio = w/h

        # a_v
        K[1][1] = h/2. / math.tan(self.camera.data.angle/2) * ratio

        # u_0
        K[0][2] = w / 2.

        # v_0
        K[1][2] = h / 2.
        K[2][2] = 1.

        return K
        '''
        camd = self.camera.data
        f_in_mm = camd.lens
        scene = bpy.context.scene
        resolution_x_in_px = scene.render.resolution_x
        resolution_y_in_px = scene.render.resolution_y
        scale = scene.render.resolution_percentage / 100
        sensor_width_in_mm = camd.sensor_width
        sensor_height_in_mm = camd.sensor_height
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        if (camd.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
            s_v = resolution_y_in_px * scale / sensor_height_in_mm
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = resolution_x_in_px * scale / sensor_width_in_mm
            s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

        # Parameters of intrinsic calibration matrix K
        alpha_u = f_in_mm * s_u
        alpha_v = f_in_mm * s_v
        u_0 = resolution_x_in_px * scale / 2
        v_0 = resolution_y_in_px * scale / 2
        skew = 0  # only use rectangular pixels

        K = Matrix(
            ((alpha_u, skew, u_0),
             (0, alpha_v, v_0),
             (0, 0, 1)))
        return K

    # Returns camera rotation and translation matrices from Blender.
    # There are 3 coordinate systems involved:
    #    1. The World coordinates: "world"
    #       - right-handed
    #    2. The Blender camera coordinates: "bcam"
    #       - x is horizontal
    #       - y is up
    #       - right-handed: negative z look-at direction
    #    3. The desired computer vision camera coordinates: "cv"
    #       - x is horizontal
    #       - y is down (to align to the actual pixel coordinates
    #         used in digital images)
    #       - right-handed: positive z look-at direction
    def compute_rotation_translation(self):
        # bcam stands for blender camera
        R_bcam2cv = Matrix(
            ((1, 0,  0),
             (0, -1, 0),
             (0, 0, -1)))

        # Transpose since the rotation is object rotation,
        # and we want coordinate rotation
        # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
        # T_world2bcam = -1*R_world2bcam * location
        #
        # Use matrix_world instead to account for all constraints
        location, rotation = self.camera.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()

        # Convert camera location to translation vector used in coordinate changes
        # T_world2bcam = -1*R_world2bcam*cam.location
        # Use location from matrix_world to account for constraints:
        T_world2bcam = -1*R_world2bcam * location

        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv = R_bcam2cv*R_world2bcam
        T_world2cv = R_bcam2cv*T_world2bcam

        # put into 3x4 matrix
        RT = Matrix((
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],)
             ))
        return RT

    def compute_projection_matrix(self):
        K = self.compute_intrinsic()
        RT = self.compute_rotation_translation()
        return K*RT, RT, K

    # # backproject pixels into 3D points
    # def backproject(self, depth):
    #     # compute projection matrix
    #     P, RT, K = self.compute_projection_matrix()
    #     P = np.matrix(P)
    #     Pinv = np.linalg.pinv(P)

    #     # compute the 3D points
    #     width = depth.shape[1]
    #     height = depth.shape[0]
    #     points = np.zeros((height, width, 3), dtype=np.float32)

    #     # camera location
    #     C = self.camera.location
    #     C = np.matrix(C).transpose()
    #     Cmat = np.tile(C, (1, width*height))

    #     # construct the 2D points matrix
    #     x, y = np.meshgrid(np.arange(width), np.arange(height))
    #     ones = np.ones((height, width), dtype=np.float32)
    #     x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    #     # backprojection
    #     x3d = Pinv * x2d.transpose()
    #     x3d[0,:] = x3d[0,:] / x3d[3,:]
    #     x3d[1,:] = x3d[1,:] / x3d[3,:]
    #     x3d[2,:] = x3d[2,:] / x3d[3,:]
    #     x3d = x3d[:3,:]

    #     # compute the ray
    #     R = x3d - Cmat

    #     # compute the norm
    #     N = np.linalg.norm(R, axis=0)

    #     # normalization
    #     R = np.divide(R, np.tile(N, (3,1)))

    #     # compute the 3D points
    #     X = Cmat + np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
    #     points[y, x, 0] = X[0,:].reshape(height, width)
    #     points[y, x, 1] = X[2,:].reshape(height, width)
    #     points[y, x, 2] = X[1,:].reshape(height, width)

    #     if 1:
    #         import matplotlib.pyplot as plt
    #         from mpl_toolkits.mplot3d import Axes3D
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         perm = np.random.permutation(np.arange(height*width))
    #         index = perm[:10000]
    #         X = points[:,:,0].flatten()
    #         Y = points[:,:,1].flatten()
    #         Z = points[:,:,2].flatten()
    #         ax.scatter(X[index], Y[index], Z[index], c='r', marker='o')
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')
    #         ax.set_zlabel('Z')
    #         ax.set_aspect('equal')
    #         plt.show()

    #     # naive way of computing the 3D points
    #     #for x in range(width):
    #     #    for y in range(height):
    #     #        if (depth[y, x] < MAX_DEPTH):
    #     #            x2d = np.matrix([x, y, 1]).transpose()
    #     #            x3d = Pinv * x2d
    #     #            x3d = x3d / x3d[3]
    #     #            x3d = x3d[:3]
    #     #            # compute the ray
    #     #            R = x3d - C
    #     #            # normalization
    #     #            R = R / np.linalg.norm(R)
    #     #            # point in 3D
    #     #            X = C + depth[y, x] * R
    #     #            # reverse y and z
    #     #            points[y, x, 0] = X[0]
    #     #            points[y, x, 1] = X[2]
    #     #            points[y, x, 2] = X[1]

    #     return points

    def render(self, image_path=os.path.join(RENDERING_PATH, 'tmp.png')):
        '''
        Render the object
        '''

        if not self.model_loaded:
            print('Model not loaded.')
            return

        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path

        bpy.ops.render.render(write_still=True)  # save straight to file\

    def vertex_paint(self, image_path=os.path.join(RENDERING_PATH, 'tmp.png'), classes=None):
        '''
        Render the object
        '''
        if not self.model_loaded:
            print('Model not loaded.')
            return

        i = 0
        for item in bpy.data.objects:
            if item.type == 'MESH':
                if item.name == 'plane':
                    mat = self.makeMaterial('transparent', (1, 1, 1), (0, 0, 0), 0)
                else:
                    item.select = True
                    # mat = bpy.data.materials.new('material_1')
                    # item.active_material = mat
                    # mat.use_vertex_color_paint = True

                    if len(item.data.vertex_colors)==0:
                        vcol_layer = item.data.vertex_colors.new()
                    else:
                        vcol_layer = item.data.vertex_colors.active

                    assert len(item.data.vertex_colors)!=0
                    vcol_layer = item.data.vertex_colors[-1]
                    item.data.vertex_colors.active = vcol_layer
                    item.data.update()

                    mat = bpy.data.materials.new('material_color_{}'.format(i))
                    i += 1

                    mat.use_vertex_color_light = False
                    mat.use_shadeless = True
                    mat.use_face_texture = False

                    #mat.diffuse_color = [1, 0, 0]
                    #mat.diffuse_shader = 'LAMBERT'
                    #mat.diffuse_intensity = 1.0
                    #mat.diffuse_color = [0, 0, 0]
                    #mat.specular_color = [0, 0, 0]
                    #mat.specular_shader = 'COOKTORR'
                    #mat.specular_intensity = 0.5
                    #mat.alpha = 1
                    #mat.ambient = 1
                    #mat.use_transparency = True
                    #mat.transparency_method = 'Z_TRANSPARENCY'
                    mat.use_vertex_color_paint = True

                    #item.data.materials.append(mat)
                    #item.active_material = mat

                if item.data.materials:
                    for i in range(len(item.data.materials)):
                        item.data.materials[i] = mat
                else:
                    item.data.materials.append(mat)
                item.active_material = mat


        # bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path
        # self.render_context.use_textures = False
        bpy.ops.render.render(write_still=True)  # save straight to file\

    def save_meta_data(self, filename):
        P, RT, K = self.compute_projection_matrix()

        meta_data = {'obj_dict': self.obj_dict,
                     'thetas': self.thetas,
                     'scales': self.scales,
                     'C': self.camera.location,
                     'projection_matrix' : np.array(P),
                     'rotation_translation_matrix': np.array(RT),
                     'intrinsic_matrix': np.array(K),
                     'azimuth': self.azimuth,
                     'elevation': self.elevation,
                     'tilt': self.tilt,
                     'distance': self.distance,
                     'viewport_size_x': self.render_context.resolution_x,
                     'viewport_size_y': self.render_context.resolution_y,
                     'camera_location': np.array(self.camera.location),
                     'factor_depth': FACTOR_DEPTH,
                     'light_info': self.light_info,
                     'environment_energy':self.environment_energy
        }

        scipy.io.savemat(filename+'.mat', meta_data)


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, xmax, ymin, ymax


def get_bbox_from_label(filename, synset_names, synset_colors):
#def get_bbox_from_label(filename, model_id, synset_names, synset_colors):
    image = imageio.imread(filename)
    print(image.shape)
    print(image.dtype)

    adata = image[:,:, -1]
    cdata = image[:,:, :3]
    colors = [np.array(color)*255 for color in synset_colors]
    print('cdata shape:',cdata.shape)

    for i, color in enumerate(colors):
        name = synset_names[i]

        np_color = np.array(color, dtype=np.uint8)
        mask = np.equal(cdata, np_color)
        mask = np.prod(mask, axis = 2)
        if np.max(mask) > 0:
            print(name, bbox(mask))

# load models, and render with different views
def render_data(renderer, data_root, name_obj, instance, arti_ind, grasp_ind, viewpoints, cam_dis, joint_angles, args=None,  _WRITE_FLAG=True, _RENDER_FLAG=True, _CREATE_FOLDER=True, RENDER_NUM=100, ARTIC_CNT=20, _RENDER_MODE='random', _USE_GUI=True, _IS_DUBUG=True):
    # >>>>>>>>>>>>>>>>>>>>>>>>>> inner setting<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
    synset_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    refer_names   = ['0', '1', '2', '3', '4', '5', '6']
    synset_scales = [1.0] * 10 # 10 is max parts
    view_num  = 5
    delta     = 10
    distance  = 1.0 # rendering distance, later to check distance?

    save_path      = '{}/{}/{}_{}_{}'.format(render_path, name_obj, instance, arti_ind, grasp_ind)
    dirname        = save_path
    hand_mesh_file = hand_mesh + '/{}/{}/{}/{}.obj'.format(name_obj, instance, arti_ind, grasp_ind)
    hand_mesh_nocs = hand_mesh + '/nocs_hand.obj'
    part_obj_path  = part_obj  +  '/{}/{}/part_objs/'.format(name_obj, instance)
    part_names    = ['none_motion.obj', 'dof_rootd_Aa001_r.obj', 'dof_rootd_Aa002_r.obj']
    num_parts     = len(part_names)
    file_paths    = [hand_mesh_file, hand_mesh_nocs]
    synset_names  = ['100', 'nocs_100']
    for i in range(len(part_names)):
        file_paths.append(part_obj_path + part_names[i])
        synset_names.append('nocs_' + refer_names[i])

    num_meshes    = len(file_paths)
    # >>>>>>>>>>>>>>>>>>>>>>>>>> setting ends here <<<<<<<<<<<<<<<<<<<<<<<<<<< #

    # create materials for masks
    materials = []
    for i in range(len(file_paths)):
        materials.append(renderer.makeMaterial('transparent', synset_colors[i], (0, 0, 0), 1))
    material_plane = renderer.makeMaterial('transparent', (1, 1, 1), (0, 0, 0), 0)

    # random sample objects until success (no collision)
    paths   = []
    scales  = []
    classes = []
    class_indexes = []

    index_all = list(range(0, num_meshes))
    # choose objects
    for index in index_all:
        scales.append(synset_scales[index])
        classes.append(synset_names[index])
        class_indexes.append(index)
        paths.append(file_paths[index])

    # load model
    instance_info = {'name_obj': name_obj, 'thetas': joint_angles, 'instance': instance, 'num_parts': 3}
    XYZlim = renderer.loadModels(paths, scales, classes, instance_info=instance_info) # 3, N, 2
    # breakpoint()
    b_max = np.max(XYZlim[:, :, 1], axis=-1)
    b_min = np.min(XYZlim[:, :, 0], axis=-1)
    center= (b_max + b_min)/2
    print('center: {}, with boundary {} {}'.format(center, b_min, b_max))

    # for i in range(1, view_num): # change to get view_num viewpoints
    #     azimuth += delta + 0.1 * np.random.randn(1)
    #     elevation += 0.1 * np.random.randn(1)
    #     tilt += 0.1 * np.random.randn(1)
    #     viewpoints[i, 0] = azimuth
    #     viewpoints[i, 1] = elevation
    #     viewpoints[i, 2] = tilt

    # render rgb images
    for i in range(view_num):
        azimuth = viewpoints[i, 0]
        elevation = viewpoints[i, 1]
        tilt = viewpoints[i, 2] # we may want to set different strategy to tilting

        # set viewpoint
        renderer.setViewpoint(azimuth, elevation, tilt, distance, 25, center=center)

        # set transparency
        renderer.setTransparency('TRANSPARENT')

        # rendering
        filename = dirname + '/rgba/%04d.png' % i
        renderer.render_context.use_textures = True
        print('---rendering rgbs image to ', filename)
        # set depth name
        renderer.fileOutput.base_path = dirname + '/depth'
        renderer.fileOutput.file_slots[0].path = '/%04d.png' % i

        renderer.render(filename)

        # save meta data
        filename = dirname + '/meta_%04d' % i
        print('---rendering meta to ', filename)
        renderer.save_meta_data(filename)


    # assign mask colors to all the materials of all the models
    for item in bpy.data.objects:
        if item.type == 'MESH':
            if item.name == 'plane':
                mat = material_plane
            else:
                ind = renderer.mesh[item.name]
                mat = materials[class_indexes[ind]]
            if item.data.materials:
                for i in range(len(item.data.materials)):
                    # item.data.materials[i] = mat
                    item.data.materials[i].diffuse_color = mat.diffuse_color
                    item.data.materials[i].diffuse_shader = mat.diffuse_shader
                    item.data.materials[i].diffuse_intensity = mat.diffuse_intensity
                    item.data.materials[i].specular_color = mat.specular_color
                    item.data.materials[i].specular_shader = mat.specular_shader
                    item.data.materials[i].specular_intensity = mat.specular_intensity
                    item.data.materials[i].alpha = mat.alpha
                    item.data.materials[i].ambient = mat.ambient
                    item.data.materials[i].use_transparency = mat.use_transparency
                    item.data.materials[i].transparency_method = mat.transparency_method
                    item.data.materials[i].use_shadeless = mat.use_shadeless
                    item.data.materials[i].use_face_texture = mat.use_face_texture
            else:
                item.data.materials.append(mat)

    # render mask
    for i in range(view_num):
        azimuth = viewpoints[i][0]
        elevation = viewpoints[i][1]
        tilt = viewpoints[i][2]

        # set viewpoint
        renderer.setViewpoint(azimuth, elevation, tilt, distance, 25, center=center)
        # set transparency
        renderer.setTransparency('TRANSPARENT')

        # rendering
        filename = dirname + '/mask/%04d.png' % i
        renderer.render_context.use_textures = False
        renderer.render(filename)

        # generate bbox
        # get_bbox_from_label(filename, synset_names, synset_colors)

    # render coordinate map
    for i in range(view_num):
        azimuth = viewpoints[i][0]
        elevation = viewpoints[i][1]
        tilt = viewpoints[i][2]

        # set viewpoint
        renderer.setViewpoint(azimuth, elevation, tilt, distance, 25, center=center)

        # set transparency
        renderer.setTransparency('TRANSPARENT')

        # rendering
        filename = dirname + '/label/%04d.png' % i
        renderer.render_context.use_textures = False
        renderer.vertex_paint(filename, classes)

    renderer.clearModel()

class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())



def main():
    # initialize the blender render
    renderer = BlenderRenderer(640, 480)
    renderer.clearModel()
    renderer._set_lighting()
    # parser = argparse.ArgumentParser()
    parser = ArgumentParserForBlender()
    parser.add_argument('--debug', action='store_true', help='indicating whether in debug mode')
    parser.add_argument('--dataset', default='shape2motion', help='name of the dataset we use')
    parser.add_argument('--item', default='eyeglasses', help='name of category we use')
    parser.add_argument('--start', default=0, help='name of category we use')
    parser.add_argument('--end', default=50, help='name of category we use')
    parser.add_argument('--dis',   default=3, help='default camera2object distance')
    parser.add_argument('--mode',  default='train', help='mode decides saving folder:train/demo')
    parser.add_argument('--roll', default='-10,10', help='camera view angle')
    parser.add_argument('--pitch', default='-90,5', help='camera view angle')
    parser.add_argument('--yaw',  default='-180,180', help='camera view angle')
    parser.add_argument('--cnt', default=30, help='count of articulation change')
    parser.add_argument('--num', default=10, help='number of rendering per articulation')
    args = parser.parse_args()
    #>>>>>>>>>>>>>>>>>>>>>>>> config end here >>>>>>>>>>>>>>>>>>>>>>>>>#

    is_debug = True
    if is_debug:
        _WRITE   = False
        _RENDER  = True
        _CREATE  = True
        _USE_GUI = True
    else:
        _WRITE   = True
        _RENDER  = True
        _CREATE  = True
        _USE_GUI = False

    num_render     = int(args.num)                                               # viewing angles
    cnt_arti       = int(args.cnt)
    cam_dis        = float(args.dis)                                             # articulation change
    name_dataset   = args.dataset

    data_root = my_dir + '/dataset/' + name_dataset
    all_ins   = os.listdir(data_root + '/urdf/' + args.item)
    all_ins.sort()
    num_instances  = len(all_ins)

    np.random.seed(5) # better to have this random seed here

    # set articulations
    angles_pre = np.load('{}/{}{}.npy'.format(whole_obj, args.item, args.item))

    # set view params
    view_file = '{}/{}.npy'.format(render_path, args.item)
    if os.path.exists(view_file):
        view_params = np.load(view_file)
    else:
        view_params = np.zeros((num_instances, cnt_arti, 10, num_render, 3))
        view_params[..., 0] = (np.random.rand(num_instances, cnt_arti, 10, num_render) - 0.5) * 240 + 90 # azimuth
        view_params[..., 1] = (np.random.rand(num_instances, cnt_arti, 10, num_render) - 0.5) * 90 # -90 is top-down, 90
        view_params[..., 2] = (np.random.rand(num_instances, cnt_arti, 10, num_render) - 0.5) * 60
        np.save(view_file , view_params)
        print('saving to ', view_file)

    if is_debug:
        instance = '0001'
        arti_ind = 0
        grasp_ind= 0
        render_data(renderer, data_root, args.item, instance, arti_ind=arti_ind, grasp_ind=grasp_ind, viewpoints=view_params[0, 0, 0], cam_dis=cam_dis, joint_angles=angles_pre[0][arti_ind], args=args, \
                            _CREATE_FOLDER=_CREATE, _WRITE_FLAG=_WRITE, _RENDER_FLAG=_RENDER, \
                            _RENDER_MODE='given', RENDER_NUM=num_render, _USE_GUI=_USE_GUI, _IS_DUBUG=is_debug)
    else:
        for i, instance in enumerate(all_ins):
            if i< args.start or i>args.end:
                continue
            for arti_ind in os.listdir(hand_mesh + '/' + args.item + '/' + instance):
                existing_objs = sorted(glob.glob( hand_mesh +'/' + args.item + '/' + instance + '/' + str(arti_ind) + '/*obj'))
                # breakpoint()
                print('go over ',  instance)
                for j, hand_urdf_file in enumerate(existing_objs):
                    state_attrs = hand_urdf_file.split('.ob')[0].split('/')[-2:]
                    arti_ind = int(state_attrs[0])
                    grasp_ind= int(state_attrs[1])
                    if grasp_ind > 12:
                        continue
                    print(i, arti_ind, grasp_ind)
                    target_path = '{}/{}/{}_{}_{}'.format(render_path, args.item, instance, arti_ind, grasp_ind)
                    if os.path.exists(target_path):
                        continue
                    render_data(renderer, data_root, args.item, instance, arti_ind=arti_ind, grasp_ind=grasp_ind, viewpoints=view_params[i, arti_ind, grasp_ind], cam_dis=cam_dis, joint_angles=angles_pre[i][arti_ind], args=args, \
                                        _CREATE_FOLDER=_CREATE, _WRITE_FLAG=_WRITE, _RENDER_FLAG=_RENDER, \
                                        _RENDER_MODE='given', RENDER_NUM=num_render, _USE_GUI=_USE_GUI, _IS_DUBUG=is_debug)
    # os.sys.exit(1)

def get_urdf(inpath, num_real_links=None):
    urdf_ins = {}
    # urdf parameters
    tree_urdf     = ET.parse(inpath + "/syn.urdf") # todo
    if num_real_links is None:
        num_real_links = len(os.listdir(inpath)) - 1 # todo
    root_urdf     = tree_urdf.getroot()
    rpy_xyz       = {}
    list_xyz      = [None] * num_real_links
    list_rpy      = [None] * num_real_links
    list_box      = [None] * num_real_links
    list_obj      = [None] * num_real_links
    # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
    num_links     = 0
    for link in root_urdf.iter('link'):
        num_links += 1
        index_link = None
        if link.attrib['name']=='base_link':
            index_link = 0
        else:
            index_link = int(link.attrib['name'])
        for visual in link.iter('visual'):
            for origin in visual.iter('origin'):
                list_xyz[index_link] = [float(x) for x in origin.attrib['xyz'].split()]
                list_rpy[index_link] = [float(x) for x in origin.attrib['rpy'].split()]
            for geometry in visual.iter('geometry'):
                for mesh in geometry.iter('mesh'):
                    list_obj[index_link] = mesh.attrib['filename']

    rpy_xyz['xyz']   = list_xyz
    rpy_xyz['rpy']   = list_rpy
    urdf_ins['link'] = rpy_xyz
    urdf_ins['obj_name'] = list_obj

    rpy_xyz       = {}
    list_xyz      = [None] * num_real_links
    list_rpy      = [None] * num_real_links
    list_axis     = [None] * num_real_links
    list_type     = [None] * num_real_links
    list_part     = [None] * num_real_links
    # here we still have to read the URDF file
    for joint in root_urdf.iter('joint'):
        index_child = int(joint.attrib['name'].split('_')[-1])
        index_parent= int(joint.attrib['name'].split('_')[0])
        list_type[index_child] = joint.attrib['type']
        list_part[index_child] = index_parent
        for origin in joint.iter('origin'):
            list_xyz[index_child] = [float(x) for x in origin.attrib['xyz'].split()]
            list_rpy[index_child] = [float(x) for x in origin.attrib['rpy'].split()]
        for axis in joint.iter('axis'):
            list_axis[index_child]= [float(x) for x in axis.attrib['xyz'].split()]
    rpy_xyz['xyz']       = list_xyz
    rpy_xyz['rpy']       = list_rpy
    rpy_xyz['axis']      = list_axis
    rpy_xyz['type']      = list_type
    rpy_xyz['parent']      = list_part

    urdf_ins['joint']    = rpy_xyz
    urdf_ins['num_links']= num_links

    return urdf_ins


if __name__ == "__main__":
    main()
    # render_all()
