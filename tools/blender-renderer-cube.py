import os
import random
import bpy
# import bpy_extras
from mathutils import Matrix, Vector
import math
import numpy as np
import scipy.io
# from scipy import misc
import imageio
# import pickle
# import png

RENDERING_PATH = './'
MAX_CAMERA_DIST = 2
MAX_DEPTH = 1e8
FACTOR_DEPTH = 0.1
g_shape_synset_name_pairs = [('02691156', 'aeroplane'),
                             ('02747177', 'ashtray'),
                             ('02773838', 'backpack'),
                             ('02801938', 'basket'),
                             ('02808440', 'tub'),  # bathtub
                             ('02818832', 'bed'),
                             ('02828884', 'bench'),
                             ('02834778', 'bicycle'),
                             ('02843684', 'mailbox'), # missing in objectnet3d, birdhouse, use view distribution of mailbox
                             ('02858304', 'boat'),
                             ('02871439', 'bookshelf'),
                             ('02876657', 'bottle'),
                             ('02880940', 'plate'), # missing in objectnet3d, bowl, use view distribution of plate
                             ('02924116', 'bus'),
                             ('02933112', 'cabinet'),
                             ('02942699', 'camera'),
                             ('02946921', 'can'),
                             ('02954340', 'cap'),
                             ('02958343', 'car'),
                             ('02992529', 'cellphone'),
                             ('03001627', 'chair'),
                             ('03046257', 'clock'),
                             ('03085013', 'keyboard'),
                             ('03207941', 'dishwasher'),
                             ('03211117', 'tvmonitor'),
                             ('03261776', 'headphone'),
                             ('03325088', 'faucet'),
                             ('03337140', 'filing_cabinet'),
                             ('03467517', 'guitar'),
                             ('03513137', 'helmet'),
                             ('03593526', 'jar'),
                             ('03624134', 'knife'),
                             ('03636649', 'lamp'),
                             ('03642806', 'laptop'),
                             ('03691459', 'speaker'),
                             ('03710193', 'mailbox'),
                             ('03759954', 'microphone'),
                             ('03761084', 'microwave'),
                             ('03790512', 'motorbike'),
                             ('03797390', 'cup'),  # missing in objectnet3d, mug, use view distribution of cup
                             ('03928116', 'piano'),
                             ('03938244', 'pillow'),
                             ('03948459', 'rifle'),  # missing in objectnet3d, pistol, use view distribution of rifle
                             ('03991062', 'pot'),
                             ('04004475', 'printer'),
                             ('04074963', 'remote_control'),
                             ('04090263', 'rifle'),
                             ('04099429', 'road_pole'),  # missing in objectnet3d, rocket, use view distribution of road_pole
                             ('04225987', 'skateboard'),
                             ('04256520', 'sofa'),
                             ('04330267', 'stove'),
                             ('04379243', 'diningtable'),  # use view distribution of dining_table
                             ('04401088', 'telephone'),
                             ('04460130', 'road_pole'),  # missing in objectnet3d, tower, use view distribution of road_pole
                             ('04468005', 'train'),
                             ('04530566', 'washing_machine'),
                             ('04554684', 'dishwasher')]  # washer, use view distribution of dishwasher

g_shape_synsets = [x[0] for x in g_shape_synset_name_pairs]
g_shape_names = [x[1] for x in g_shape_synset_name_pairs]
g_view_distribution_files = dict(zip(g_shape_synsets, [name+'.txt' for name in g_shape_names]))

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



    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov):

        cx, cy, cz = obj_centered_camera_pos(distance_ratio * MAX_CAMERA_DIST, azimuth, altitude)
        q1 = camPosToQuaternion(cx, cy, cz)
        q2 = camRotQuaternion(cx, cy, cz, yaw)
        q = quaternionProduct(q2, q1)

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
        mat.diffuse_color = diffuse
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


    def loadModels(self, file_paths, scales, classes, filename):
        self.model_loaded = True


        mesh = dict()

        num = len(file_paths)
        height_max = -np.inf * np.ones((num,), dtype=np.float32)
        height_min = np.inf * np.ones((num,), dtype=np.float32)

        for i in range(num):
            file_path = file_paths[i]
            try:
                if file_path.endswith('obj'):
                    bpy.ops.import_scene.obj(filepath=file_path)
                elif file_path.endswith('3ds'):
                    bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
                elif file_path.endswith('dae'):
                    # Must install OpenCollada. Please read README.md for installation
                    bpy.ops.wm.collada_import(filepath=file_path)
                else:
                    # TODO
                    # Other formats not supported yet
                    self.model_loaded = False
                    raise Exception("Loading failed: %s" % (file_path))

                for item in bpy.data.objects:
                    if item.type == 'MESH':
                        if item.name not in mesh:
                            mesh[item.name] = i
                            for vertex in item.data.vertices:
                                height_max[i] = max(height_max[i], scales[i] * vertex.co[1])
                                height_min[i] = min(height_min[i], scales[i] * vertex.co[1])

            except Exception:
                self.model_loaded = False
        self.mesh = mesh
        print(height_max)
        print(height_min)

        # add new_material
        for item in bpy.data.objects:
            if item.type == 'MESH':
                vcol_layer = item.data.vertex_colors.new()
                for loop_index, loop in enumerate(item.data.loops):
                    loop_vert_index = loop.vertex_index

                    # given that the objects from ShapeNetCoreV1 are initially centered and normalized by bbox diagonal
                    # the upper limits of the scopes of x, y, z coordinates will all be [-0.5, 0.5]
                    color = item.data.vertices[loop_vert_index].co + Vector([0.5, 0.5, 0.5])

                    print(color)
                    vcol_layer.data[loop_index].color = color



        # rotate the objects
        thetas = np.zeros((num,), dtype=np.float32)
        translations = np.zeros((num, 3), dtype=np.float32)


        for i in range(1, num):
            # sample a rotation angle
            thetas[i] = (2 * np.random.rand(1) - 1) * math.pi

        for item in bpy.data.objects:
            if item.type == 'MESH':
                ind = mesh[item.name]
                theta = thetas[ind]
                
                R = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
                for vertex in item.data.vertices:
                    rv = np.dot(R, np.array(vertex.co).reshape((3,1)))
                    vertex.co[0] = rv[0]
                    vertex.co[1] = rv[1]
                    vertex.co[2] = rv[2]
                    # os.sys.exit(1)

        
        # rescale the meshes
        for item in bpy.data.objects:
            if item.type == 'MESH':
                ind = mesh[item.name]
                for vertex in item.data.vertices:
                    vertex.co *= scales[ind]



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
        # make sure table, chair and sofa are on the ground
        table_height_min = height_min[0]
        for item in bpy.data.objects:
            if item.type == 'MESH':
                ind = mesh[item.name]
                if classes[ind] == 'chair' or classes[ind] == 'sofa':
                    for vertex in item.data.vertices:
                        vertex.co[1] += table_height_min - height_min[ind]
                    translations[ind, 1] += table_height_min - height_min[ind]

        # move objects on the table
        table_height_max = height_max[0]
        for item in bpy.data.objects:
            if item.type == 'MESH':
                ind = mesh[item.name]
                if classes[ind] != 'chair' and classes[ind] != 'sofa' and classes[ind] != 'table':
                    for vertex in item.data.vertices:
                        vertex.co[1] += table_height_max - height_min[ind]
                    translations[ind, 1] += table_height_max - height_min[ind]

        # move all the objects down
        for item in bpy.data.objects:
            if item.type == 'MESH':
                ind = mesh[item.name]
                for vertex in item.data.vertices:
                    vertex.co[1] -= 0.2
                translations[ind, 1] -= 0.2

        # collect the vertices
        vertices = []
        for i in range(num):
            vertices.append(np.zeros((0, 3), dtype=np.float32))
        for item in bpy.data.objects:
            if item.type == 'MESH':
                ind = mesh[item.name]
                for vertex in item.data.vertices:
                    vertices[ind] = np.append(vertices[ind], np.array(vertex.co).reshape((1,3)), axis = 0)

        # compute the boundary of objects
        Xlim = np.zeros((num, 2), dtype=np.float32)
        Ylim = np.zeros((num, 2), dtype=np.float32)
        Zlim = np.zeros((num, 2), dtype=np.float32)
        for i in range(num):
            Xlim[i, 0] = vertices[i][:, 0].min()
            Xlim[i, 1] = vertices[i][:, 0].max()
            Ylim[i, 0] = vertices[i][:, 1].min()
            Ylim[i, 1] = vertices[i][:, 1].max()
            Zlim[i, 0] = vertices[i][:, 2].min()
            Zlim[i, 1] = vertices[i][:, 2].max()

        # sampling locations of objects
        locations = np.zeros((num, 2), dtype=np.float32)
        success = True
        for i in range(1, num):
            if classes[i] == 'chair' or classes[i] == 'sofa':
                table_top = False
            else:
                table_top = True
            count = 0
            while 1:
                count += 1
                if table_top:
                    lx = Xlim[0, 1] - Xlim[0, 0]
                    lz = Zlim[0, 1] - Zlim[0, 0]
                    x = Xlim[0, 0] + np.random.rand(1) * lx
                    z = Zlim[0, 0] + np.random.rand(1) * lz
                    # check if object is inside the table or not
                    a = [x-(Xlim[i,1]-Xlim[i,0])/2, z-(Zlim[i,1]-Zlim[i,0])/2, x+(Xlim[i,1]-Xlim[i,0])/2, z+(Zlim[i,1]-Zlim[i,0])/2]
                    if a[2] < Xlim[0, 1] - 0.02 and \
                       a[0] > Xlim[0, 0] + 0.02 and \
                       a[3] < Zlim[0, 1] - 0.02 and \
                       a[1] > Zlim[0, 0] + 0.02:
                        flag = 1
                    else:
                        flag = 0
                else:
                    lx = Xlim[0, 1] - Xlim[0, 0]
                    lz = Zlim[0, 1] - Zlim[0, 0]
                    if i == 1:
                        x = Xlim[0, 0] - 0.5
                        z = Zlim[0, 0] + np.random.rand(1) * lz
                    elif i == 2:
                        x = Xlim[0, 1] + 0.5
                        z = Zlim[0, 0] + np.random.rand(1) * lz
                    elif i == 3:
                        x = Xlim[0, 0] + np.random.rand(1) * lx
                        z = Zlim[0, 0] - 0.5
                    elif i == 4:
                        x = Xlim[0, 0] + np.random.rand(1) * lx
                        z = Zlim[0, 1] + 0.5
                    a = [x-(Xlim[i,1]-Xlim[i,0])/2, z-(Zlim[i,1]-Zlim[i,0])/2, x+(Xlim[i,1]-Xlim[i,0])/2, z+(Zlim[i,1]-Zlim[i,0])/2]
                    flag = 1

                if flag == 1:
                    # check collision with other objects
                    if table_top:
                        r = range(1, i)
                    else:
                        r = range(0, i)
                    for j in r:
                        b = [locations[j,0]-(Xlim[j,1]-Xlim[j,0])/2, locations[j,1]-(Zlim[j,1]-Zlim[j,0])/2, \
                             locations[j,0]+(Xlim[j,1]-Xlim[j,0])/2, locations[j,1]+(Zlim[j,1]-Zlim[j,0])/2]
                        x1 = max(a[0], b[0])
                        y1 = max(a[1], b[1]);
                        x2 = min(a[2], b[2]);
                        y2 = min(a[3], b[3]);
                        w = x2 - x1;
                        h = y2 - y1;
                        inter = w * h
                        if w > 0 and h > 0 and inter > 0:
                            print('object {:d} collision with object {:d}'.format(i, j))
                            flag = 0

                if flag == 1:
                    print('Sampled location for object %d' % i)
                    break
                else:
                    if count > 1000:
                        print('Fail: cannot find location for object %d' % i)
                        break

            if flag == 1:
                locations[i, 0] = x
                locations[i, 1] = z
            else:
                success = False
                break
        
        if success:
            # move the meshes



            for item in bpy.data.objects:
                if item.type == 'MESH':
                    ind = mesh[item.name]
                    if ind > 0:
                        for vertex in item.data.vertices:
                            vertex.co[0] += locations[ind, 0]
                            vertex.co[2] += locations[ind, 1]
                        translations[ind, 0] += locations[ind, 0]
                        translations[ind, 2] += locations[ind, 1]

        # add a transparent plane
        V = np.zeros((0, 3), dtype=np.float32)
        for item in bpy.data.objects:
            if item.type == 'MESH':
                for vertex in item.data.vertices:
                    V = np.append(V, np.array(vertex.co).reshape((1,3)), axis = 0)

        factor = 3
        x1 = factor * np.min(V[:,0])
        x2 = factor * np.max(V[:,0])
        y1 = factor * np.min(V[:,2])
        y2 = factor * np.max(V[:,2])
        z = np.min(V[:,1])

        verts = [(x1, y1, z), (x2, y1, z), (x2, y2, z), (x1, y2, z)]
        faces = [(0, 1, 2, 3)]

        mesh_data = bpy.data.meshes.new("cube_mesh_data")
        obj = bpy.data.objects.new("plane", mesh_data)
        bpy.context.scene.objects.link(obj)
        bpy.context.scene.objects.active = obj
        obj.select = True

        mesh_data.from_pydata(verts, [], faces)  
        mesh_data.update()

        mat = self.makeMaterial('transparent', (0.5,0.5,0.5), (0,0,0), 1)
        obj.data.materials.append(mat)

        if success:
            # save model
            self.obj_dict = mesh
            self.thetas = thetas
            self.translations = translations
            self.scales = scales

            if filename:
                bpy.ops.export_scene.obj(filepath=filename, use_selection=False)

        return success


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

    # backproject pixels into 3D points
    def backproject(self, depth):
        # compute projection matrix
        P, RT, K = self.compute_projection_matrix()
        P = np.matrix(P)
        Pinv = np.linalg.pinv(P)

        # compute the 3D points        
        width = depth.shape[1]
        height = depth.shape[0]
        points = np.zeros((height, width, 3), dtype=np.float32)

        # camera location
        C = self.camera.location
        C = np.matrix(C).transpose()
        Cmat = np.tile(C, (1, width*height))

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

        # backprojection
        x3d = Pinv * x2d.transpose()
        x3d[0,:] = x3d[0,:] / x3d[3,:]
        x3d[1,:] = x3d[1,:] / x3d[3,:]
        x3d[2,:] = x3d[2,:] / x3d[3,:]
        x3d = x3d[:3,:]

        # compute the ray
        R = x3d - Cmat

        # compute the norm
        N = np.linalg.norm(R, axis=0)
        
        # normalization
        R = np.divide(R, np.tile(N, (3,1)))

        # compute the 3D points
        X = Cmat + np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
        points[y, x, 0] = X[0,:].reshape(height, width)
        points[y, x, 1] = X[2,:].reshape(height, width)
        points[y, x, 2] = X[1,:].reshape(height, width)

        if 0:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            perm = np.random.permutation(np.arange(height*width))
            index = perm[:10000]
            X = points[:,:,0].flatten()
            Y = points[:,:,1].flatten()
            Z = points[:,:,2].flatten()
            ax.scatter(X[index], Y[index], Z[index], c='r', marker='o')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_aspect('equal')          
            plt.show()

        # naive way of computing the 3D points
        #for x in range(width):
        #    for y in range(height):
        #        if (depth[y, x] < MAX_DEPTH):
        #            x2d = np.matrix([x, y, 1]).transpose()
        #            x3d = Pinv * x2d
        #            x3d = x3d / x3d[3]
        #            x3d = x3d[:3]
        #            # compute the ray
        #            R = x3d - C
        #            # normalization
        #            R = R / np.linalg.norm(R)
        #            # point in 3D
        #            X = C + depth[y, x] * R
        #            # reverse y and z
        #            points[y, x, 0] = X[0]
        #            points[y, x, 1] = X[2]
        #            points[y, x, 2] = X[1]

        return points

            
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


    def vertex_paint(self, image_path=os.path.join(RENDERING_PATH, 'tmp.png')):
        '''
        Render the object
        '''
        if not self.model_loaded:
            print('Model not loaded.')
            return

        #current_obj = bpy.context.active_object
        #mesh = current_obj.data

        i = 0
        for item in bpy.data.objects:
            print(item)
            if item.type == 'MESH':
                #bpy.context.scene.item.select = Truebjects.active = item
                #bpy.context.active_object = item
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

            else:
                print(item, item.type)

        #bpy.ops.object.mode_set(mode='VERTEX_PAINT')

        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # save straight to file\


    def save_meta_data(self, filename):
        P, RT, K = self.compute_projection_matrix()

        meta_data = {'obj_dict': self.obj_dict,
                     'thetas': self.thetas,
                     'translations': self.translations,
                     'scales': self.scales,
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

def render_all():
    '''Test function'''

    # table, chair, sofa, 'lamp', tvmonitor, bottle, mug, bowl, can, keyboard, cap
    synsets = ['04379243', '03001627', '04256520', '03636649', '03211117', '02876657', '03797390', '02880940', '02946921', '03085013', '02954340']
    #shapenet_root = '/var/Projects/ShapeNetCore.v1'
    #results_root = '/var/Projects/Deep_ISM/Rendering/images'
    shapenet_root = '/orions4-zfs/projects/hewang/ShapeNetCore.v1'
    results_root = '/orions4-zfs/projects/hewang/shapenet_scene/new_data'
    if not os.path.exists(results_root):
        os.makedirs(results_root)

    # load 3D shape paths
    file_paths = []
    model_ids = []

    for i in range(len(synsets)):
        synset = synsets[i]
        dn = os.path.join(shapenet_root, synset)
        model_id = [line.strip('\n') for line in open(dn + '/models.txt')]
        file_paths.append( [os.path.join(dn, line, 'model.obj') for line in model_id] )
        model_ids.append(model_id)

    # initialize the blender render
    renderer = BlenderRenderer(640, 480)

    for i in range(len(synsets)):
        synset = synsets[i]
        dirname = os.path.join(results_root, synset)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        file_path = file_paths[i]
        model_id = model_ids[i]
        for j in range(len(model_id)):
            id = model_id[j]
            n = len(id)
            id = id[:n-1]
            
            renderer.loadModel(file_path[j])

            # set viewpoint
            renderer.setViewpoint(45, 30, 0, 0.6, 25)

            # set transparency
            renderer.setTransparency('TRANSPARENT')

            # rendering
            filename = dirname + '/' + id + '.png'
            print(filename)
            renderer.render_context.use_textures = True
            renderer.render(filename)

            renderer.clearModel()

    os.sys.exit(1)


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



def main():
    '''Test function'''

    synsets = ['04379243', '03211117', '02876657', '03797390', '02946921', '03085013', '02954340']
    synset_names = ['table', 'tvmonitor', 'bottle', 'mug', 'can', 'keyboard', 'cap']
    synset_scales = [1.0, 0.4, 0.2, 0.2, 0.2, 0.4, 0.3]
    synset_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    #num_scene = 100
    #view_num = 100
    num_scene = 1
    view_num = 1
    delta = 2.5
    distance = 0.6

    shapenet_root = '/orions4-zfs/projects/hewang/ShapeNetCore.v1'
    models_root = '/orions4-zfs/projects/hewang/shapenet_scene/images_selected'
    results_root = '/orions4-zfs/projects/hewang/shapenet_scene/new_data'
    view_dists_root = '/orions4-zfs/projects/hewang/shapenet_scene/view_distributions'
    if not os.path.exists(results_root):
        os.makedirs(results_root)

    # load 3D shape paths
    file_paths = []
    for i in range(len(synsets)):
        synset = synsets[i]
        dn = os.path.join(shapenet_root, synset)
        model_dir = os.path.join(models_root, synset)
        model_id = [line.strip('\n') for line in open(model_dir + '/models.txt')]
        file_paths.append( [os.path.join(dn, line, 'model.obj') for line in model_id] )

    # load viewpoint distributions
    synset = synsets[0] # table
    filename = os.path.join(view_dists_root, g_view_distribution_files[synset])
    if not os.path.exists(filename):
        print('Failed to read view distribution files from %s for synset %s' % 
              (filename, synset))
        exit()
    view_params = open(filename).readlines()
    view_params = [[float(x) for x in line.strip().split(' ')] for line in view_params]

    # initialize the blender render
    renderer = BlenderRenderer(640, 480)

    # for each scene
    for k in range(num_scene):

        # create output directory
        dirname = os.path.join(results_root, '%04d' % k)
        renderer.dirname = dirname


        print(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = dirname + '/model.obj'
        '''
        if os.path.exists(filename):
            bpy.ops.import_scene.obj(filepath=filename)
            renderer.model_loaded = True

            meta_name = dirname + '/%04d_meta.mat' % 0
            meta_data = scipy.io.loadmat(meta_name)
            light_info = meta_data['light_info']
            environment_energy = meta_data['environment_energy']
            renderer._set_lighting(light_info, environment_energy)
        '''

        renderer._set_lighting()

        paths = []
        scales = []
        classes = []
        class_indexes = []

        # create materials
        materials = []
        for i in range(len(synsets)):
            materials.append(renderer.makeMaterial('transparent', synset_colors[i], (0, 0, 0), 1))
        material_plane = renderer.makeMaterial('transparent', (1, 1, 1), (0, 0, 0), 0)
        # random sample objects until success (no collision)
        while 1:
            # choose a table
            id = 0 ## synset for table
            model_id = file_paths[id]

            # guarantee that no table in training dataset is seen in the test dataset
            if k < num_scene / 2:
                index = random.randint(0, int(len(model_id) / 2))
            else:
                index = random.randint(int(len(model_id) / 2) + 1, len(model_id)-1)

            # specify scale, class, class_index of table
            # synset_names = ['table', 'tvmonitor', 'bottle', 'mug', 'can', 'keyboard', 'cap']
            # synset_scales = [1.0, 0.4, 0.2, 0.2, 0.2, 0.4, 0.3]
            # synset_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]

            paths.append(model_id[index])
            scales.append(synset_scales[id])
            classes.append(synset_names[id])
            class_indexes.append(id)

            # choose the number of objects on the table
            num = 5
            index_all = np.random.permutation(6) + 1 # e.g. [1, 6, 2, 3, 4, 5],  without 0 (table)

            # choose objects
            for i in range(num):
                index = index_all[i]
                model_id = file_paths[index]
                scales.append(synset_scales[index])
                classes.append(synset_names[index])
                class_indexes.append(index)

                # guarantee that no table in training dataset is seen in the test dataset
                if k < num_scene / 2:
                    index = random.randint(0, int(len(model_id) / 2))
                else:
                    index = random.randint(int(len(model_id) / 2) + 1, len(model_id)-1)
                paths.append(model_id[index])

            print(classes)

            # load model

            success = renderer.loadModels(paths, scales, classes, filename)

            if success:
                break
            else:
                paths = []
                scales = []
                classes = []
                class_indexes = []
                renderer.clearModel()

                renderer._set_lighting()

                # create materials
                materials = []
                for i in range(len(synsets)):
                    materials.append(renderer.makeMaterial('transparent', synset_colors[i], (0,0,0), 1))
                material_plane = renderer.makeMaterial('transparent', (1,1,1), (0,0,0), 0)



        # sample viewpoints

        #if os.path.exists(filename):
        #    bpy.ops.import_scene.obj(filepath=filename)
        #    renderer.model_loaded = True


        viewpoints = np.zeros((view_num, 3), dtype=np.float32)
        while 1:
            index = random.randint(0, len(view_params)-1)
            azimuth = view_params[index][0]
            elevation = view_params[index][1]
            tilt = view_params[index][2]
            tilt = 0
            if elevation > 30 and elevation < 40:
                break
        viewpoints[0, 0] = azimuth
        viewpoints[0, 1] = elevation
        viewpoints[0, 2] = tilt
        for i in range(1, view_num):
            azimuth += delta + 0.1 * np.random.randn(1)
            elevation += 0.1 * np.random.randn(1)
            tilt += 0.1 * np.random.randn(1)
            viewpoints[i, 0] = azimuth
            viewpoints[i, 1] = elevation
            viewpoints[i, 2] = tilt



        # render rgb images
        for i in range(view_num):
            azimuth = viewpoints[i, 0]
            elevation = viewpoints[i, 1]
            tilt = viewpoints[i, 2]

            # set viewpoint
            renderer.setViewpoint(azimuth, elevation, tilt, distance, 25)

            # set transparency
            renderer.setTransparency('TRANSPARENT')

            # rendering
            filename = dirname + '/%04d_rgba.png' % i
            renderer.render_context.use_textures = True

            # set depth name
            renderer.fileOutput.base_path = dirname
            renderer.fileOutput.file_slots[0].path = '%04ddepth#.png' % i

            renderer.render(filename)

            # save meta data
            filename = dirname + '/%04d_meta' % i
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
            renderer.setViewpoint(azimuth, elevation, tilt, distance, 25)

            # set transparency
            renderer.setTransparency('TRANSPARENT')

            # rendering
            filename = dirname + '/%04d_mask.png' % i
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
            renderer.setViewpoint(azimuth, elevation, tilt, distance, 25)

            # set transparency
            renderer.setTransparency('TRANSPARENT')

            # rendering
            filename = dirname + '/%04d_label.png' % i
            renderer.render_context.use_textures = False
            renderer.vertex_paint(filename)

        renderer.clearModel()
    os.sys.exit(1)


if __name__ == "__main__":
    main()
    # render_all()
