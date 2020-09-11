"""
func: parse articulation infos from json into URDF files used by Pybullet
    - traverse over path and objects
    - parse from json file(dict, list, array);
    - obj files path;
    - inverse computing on rpy/xyz;
    - xml writer to urdf
"""
import os
import os.path
import glob
import sys
import time
import json
import copy
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, SubElement, Comment, ElementTree, XML
import xml.dom.minidom

import _init_paths
from global_info import global_info

def breakpoint():
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    #>>>>>>>>>>>>>>>> you only need to change this part >>>>>>>>>>>>>
    is_debug  = False
    dataset   = 'shape2motion'
    infos     = global_info()
    my_dir    = infos.base_path
    urdf_path = infos.urdf_path
    base_path = infos.hand_path # with a lot of different objects, and then 0001/0002/0003/0004/0005
    #>>>>>>>>>>>>>>>>>>>>>>>> config end >>>>>>>>>>>>>>>>>>>>>>>>>>>>#

    all_objs     = os.listdir( base_path )
    print('all_objs are: ', all_objs)
    object_nums = []
    object_joints = []
    # with open(base_path + '/'  '/statistics.txt', "a+") as f:
    #     for obj_n in all_objs:
    #         f.write('{}\t'.format(obj_n))
    #     f.write('\n')
    for obj_name in all_objs:
        for instance in os.listdir(base_path  + '/' + obj_name):
            for arti_ind in os.listdir(base_path  + '/' + obj_name + '/' + instance):
                hand_per_articulation = sorted(glob.glob(base_path  + '/' + obj_name + '/' + instance + '/' + arti_ind + '/*obj'))
                obj_num =len(hand_per_articulation)
                object_nums.append(obj_num)
                if is_debug:
                    selected_objs = hand_per_articulation[0:1]
                else:
                    selected_objs = hand_per_articulation

                for hand_obj in selected_objs:                        # regular expression
                    #
                    grasp_ind    = hand_obj.split('.ob')[0].split('/')[-1]
                    save_dir     = urdf_path  + '/' + obj_name + '/' + instance + '/' + arti_ind 

                    keys_link = ['dof_rootd']    # +  list(joint_dict.keys()) #

                    #>>>>>>>>>>>>>>>>>> contruct links and joints
                    root  = Element('robot', name="block")
                    num   = 1
                    links_name = ["base_link"] # + [str(i+1) for i in range(num)]
                    all_kinds_joints = ["revolute", "fixed", "prismatic", "continuous", "planar"]
                    joints_name = []
                    joints_type = []
                    joints_pos  = []
                    links_pos   = [None] * num
                    joints_axis = []
                    # parts connection
                    rotation_joint = 0
                    translation_joint = 0

                    children = [
                        Element('link', name=links_name[i])
                        for i in range(num)
                        ]
                    joints = [
                        Element('joint', name=joints_name[i], type=joints_type[i])
                        for i in range(num-1)
                        ]
                    # add inertial component
                    node_inertial = XML('''<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="1.0"/><inertia ixx="0.9" ixy="0.9" ixz="0.9" iyy="0.9" iyz="0" izz="0.9"/></inertial>''')
                    #>>>>>>>>>>>. 1. links
                    for i in range(num):
                        visual   = SubElement(children[i], 'visual')
                        dof_name = keys_link[i]
                        if dof_name == 'dof_rootd':
                            origin   = SubElement(visual, 'origin', rpy="0.0 0.0 0.0", xyz="0 0 0")
                        # else:
                        #     origin   = SubElement(visual, 'origin', rpy="0.0 0.0 0.0", xyz="{} {} {}".format(links_pos[i][0], links_pos[i][1], links_pos[i][2]))
                        geometry = SubElement(visual, 'geometry')
                        mesh     = SubElement(geometry, 'mesh', filename=hand_obj)
                        # else:
                        #     mesh     = SubElement(geometry, 'mesh', filename="{}/part_objs/{}.obj".format(sub_dir, dof_name))
                        # materials assignment
                        inertial = SubElement(children[i], 'inertial')
                        node_inertial = XML('''<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="3.0"/><inertia ixx="100" ixy="100" ixz="100" iyy="100" iyz="100" izz="100"/></inertial>''')
                        inertial.extend(node_inertial)
                        if i == 0:
                            for mass in inertial.iter('mass'):
                                mass.set('value', "0.0")
                            for inertia in inertial.iter('inertia'):
                                inertia.set('ixx', "0.0")
                                inertia.set('ixy', "0.0")
                                inertia.set('ixz', "0.0")
                                inertia.set('iyy', "0.0")
                                inertia.set('iyz', "0.0")
                                inertia.set('izz', "0.0")

                    #>>>>>>>>>>>> 3. construct the trees
                    root.extend(children)
                    if len(joints)>0:
                        root.extend(joints)
                    xml_string = xml.dom.minidom.parseString(tostring(root))
                    xml_pretty_str = xml_string.toprettyxml()
                    # breakpoint()
                    # print(xml_pretty_str)

                    tree = ET.ElementTree(root)
                    # save
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    with open(save_dir + f'/{grasp_ind}.urdf', "w") as f:
                        print('writing to ', save_dir + f'/{grasp_ind}.urdf')
                        f.write(xml_pretty_str)
                    # #>>>>>>>>>>>>>>>>> coding >>>>>>>>>>>>>>
                    # # Create a copy
                    # for i in range(num):
                    #     member_part = copy.deepcopy(root)
                    #     # remove all visual nodes directly
                    #     for link in member_part.findall('link'):
                    #         if link.attrib['name']!=links_name[i]:
                    #             for visual in link.findall('visual'):
                    #                 link.remove(visual)
                    #     xml_string = xml.dom.minidom.parseString(tostring(member_part))
                    #     xml_pretty_str = xml_string.toprettyxml()
                    #     tree = ET.ElementTree(member_part)
                    #     with open(save_dir + '/syn_p{}.urdf'.format(i), "w") as f:
                    #         f.write(xml_pretty_str)
    #     object_joints.append(object_joints_per)
    # with open(base_path + '/'  '/statistics.txt', "a+") as f:
    #     for obj_num in object_nums:
    #         f.write('{}\t'.format(obj_num))
    #     f.write('\n')
    # with open(base_path + '/'  '/statistics.txt', "a+") as f:
    #     for obj_j in object_joints:
    #         f.write('{}/{}\t'.format(obj_j['revolute'], obj_j['prismatic']))
    #     f.write('\n')
