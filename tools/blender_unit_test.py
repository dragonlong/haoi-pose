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

#>>>>>>>>>>>>>>>>>>>>>>>> global <<<<<<<<<<<<<<<<<<<<<<<<<<<<#
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
#>>>>>>>>>>>>>>>>>>>>>>>> ends here <<<<<<<<<<<<<<<<<<<<<<<<<#

def breakpoint():
    import pdb;pdb.set_trace()

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class active: 
    # operations on active objects
    def rename(objName):
        bpy.context.object.name = objName

class select:
    # operations on selected objects
    # Declarative
    def scale(objName, v):
        bpy.data.objects[objName].scale = v
    # Declarative
    def location(objName, v):
        bpy.data.objects[objName].location = v
    # Declarative
    def rotation(objName, v):
        bpy.data.objects[objName].rotation_euler = v

# create primitives
class create:
    """Function Class for CREATING Objects"""
    def cube(objName, v=0.5):
        bpy.ops.mesh.primitive_cube_add(radius=v, location=(0, 0, 0))
        active.rename(objName)
    def sphere(objName, v=0.5):
        bpy.ops.mesh.primitive_uv_sphere_add(size=v, location=(0, 0, 0))
        active.rename(objName)
    def cone(objName, v=0.5):
        bpy.ops.mesh.primitive_cone_add(radius1=v, location=(0, 0, 0))
        active.rename(objName)

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

def obj_lookat_positioned_camera_pos(dist, azimuth_deg, elevation_deg):
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
        v = tree.nodes.new('CompositorNodeViewer')
        links.new(rl.outputs[2], v.inputs[0])  # link Image output to Viewer input

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

                lx, ly, lz = obj_lookat_positioned_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
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
                lx, ly, lz = obj_lookat_positioned_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
                bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
                light_energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
                bpy.data.objects['Point'].data.energy = light_energy

                light_info[i, 0] = light_azimuth_deg
                light_info[i, 1] = light_elevation_deg
                light_info[i, 2] = light_dist
                light_info[i, 3] = light_energy

        self.environment_energy = bpy.context.scene.world.light_settings.environment_energy
        self.light_info = light_info


    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov, lookat_position=None):

        cx, cy, cz = obj_lookat_positioned_camera_pos(distance_ratio * MAX_CAMERA_DIST, azimuth, altitude)
        q1 = camPosToQuaternion(cx, cy, cz)
        q2 = camRotQuaternion(cx, cy, cz, yaw)
        q = quaternionProduct(q2, q1)

        if lookat_position is not None:
            self.camera.location[0] = cx + lookat_position[0]
            self.camera.location[1] = cy + lookat_position[1]
            self.camera.location[2] = cz + lookat_position[2]
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

    def render(self, image_path=os.path.join(RENDERING_PATH, 'tmp.png')):
        '''
        Render the object
        '''
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
        print('entering NOCS rendering mode')
        i = 0
        for item in bpy.data.objects:
            if item.type == 'MESH':
                if item.name == 'plane':
                    mat = self.makeMaterial('transparent', (1, 1, 1), (0, 0, 0), 0)
                else:
                    item.select = True
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
                    mat.use_vertex_color_paint = True

                    # mat.diffuse_color = [0, 0, 0]
                    # mat.diffuse_shader = 'LAMBERT'
                    # mat.diffuse_intensity = 1.0
                    # mat.specular_color = [0, 0, 0]
                    # mat.specular_shader = 'COOKTORR'
                    # mat.specular_intensity = 0.5
                    # mat.alpha = 1
                    # mat.ambient = 1
                    # mat.use_transparency = True
                    # mat.transparency_method = 'Z_TRANSPARENCY'
                
                if item.data.materials:
                    for i in range(len(item.data.materials)):
                        item.data.materials[i] = mat
                else:
                    item.data.materials.append(mat)
                item.active_material = mat

            
        # bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path
        self.render_context.use_textures = False
        bpy.ops.render.render(write_still=True)  # save straight to file\

def main():
    # initialize the blender render
    renderer = BlenderRenderer(640, 480)
    renderer.clearModel()
    renderer._set_lighting()
    # 
    dirname = './'

    # create a Cube
    create.cube('PerfectCube')
    renderer.model_loaded = True
    
    # material for masks
    materials = []
    synset_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    for i in range(len(synset_colors)):
        materials.append(renderer.makeMaterial('transparent', synset_colors[i], (0, 0, 0), 1))

    # set camera view points 
    view_num = 1
    distance = 2.0
    lookat_position = [0, 0, 0]
    viewpoints = np.zeros((view_num, 3))
    viewpoints[..., 0] = (np.random.rand(view_num) - 0.5) * 0 + 45 # azimuth
    viewpoints[..., 1] = (np.random.rand(view_num) - 0.5) * 0 + 45 # elevation
    viewpoints[..., 2] = (np.random.rand(view_num) - 0.5) * 0 # tilt

    # assign vertex colors by coordinates
    for item in bpy.data.objects:
        if item.type == 'MESH': 
            item.select = True
            if len(item.data.vertex_colors)==0:
                print('---using material from ', item.name)
                vcol_layer = item.data.vertex_colors.new()
                for loop_index, loop in enumerate(item.data.loops):
                    loop_vert_index = loop.vertex_index
                    color = item.data.vertices[loop_vert_index].co
                    # color = Vector([0.5, 0.5, 0.5])
                    # color = (item_color.data.vertices[loop_vert_index].co - Vector(XYZ_c[:, mesh[item_color.name]].tolist())) / XYZ_l[mesh[item_color.name]] + Vector([0.5, 0.5, 0.5])
                    vcol_layer.data[loop_index].color = color
            else:
                vcol_layer = item.data.vertex_colors.active
            item.select = False

    # assign mask colors to all the materials of all the models
    for item in bpy.data.objects:
        if item.type == 'MESH':
            mat = materials[0]
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
        renderer.setViewpoint(azimuth, elevation, tilt, distance, 25, lookat_position=lookat_position)
        # set transparency
        renderer.setTransparency('TRANSPARENT')

        # rendering
        filename = dirname + '/mask/%04d.png' % i
        renderer.render_context.use_textures = False
        renderer.render(filename)

    # render coordinate map
    for i in range(view_num):
        azimuth = viewpoints[i][0]
        elevation = viewpoints[i][1]
        tilt = viewpoints[i][2]

        # set viewpoint
        renderer.setViewpoint(azimuth, elevation, tilt, distance, 25, lookat_position=lookat_position)

        # set transparency
        renderer.setTransparency('TRANSPARENT')

        # rendering
        filename = dirname + '/label/%04d.png' % i
        renderer.render_context.use_textures = False
        renderer.vertex_paint(filename)
    # renderer.clearModel()

if __name__ == "__main__":
    main()
