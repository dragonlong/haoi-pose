roscore
conda deactivate
conda deactivate

export GRASPIT=~/.graspit
export GRASPIT_PLUGIN_DIR='/home/dragon/graspit_ros_ws/devel/lib/' # libgraspit_interface.so
===========>> graspit <<========

===========>> ROS setup <=======
# sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
# sudo apt install python-rosdep
# sudo apt-get install python-catkin-tools
# sudo rosdep init
# rosdep update

# echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
# source ~/.bashrc
sudo chmod -R a+rwx ~/.graspit/
export GRASPIT=~/.graspit
export GRASPIT_PLUGIN_DIR='/home/dragon/graspit_ros_ws/devel/lib/'
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

==========>> graspit-interface ======
# in graspit_interface
source /home/dragon/graspit_ros_ws/devel/setup.bash
roslaunch graspit_interface graspit_interface.launch

rostopic list
rosservice list

=========>> mano_grasp ===========
hand
cd mano_grasp
python setup.py install --user --graspit_dir=$GRASPIT
# /home/dragon/.local/lib/python2.7/site-packages/

========== usage =================
python -m mano_grasp.prepare_objects --models_folder /home/dragon/Downloads/ICML2021/YCB_Affordance/data --file_out YCB_DATASET_objects.txt  --scales 1000
python -m mano_grasp.prepare_objects --models_folder /home/dragon/Documents/CVPR2020/dataset/shape2motion/objects --file_out shape2motion_eyeglasses.txt
python -m mano_grasp.prepare_objects --models_folder /home/dragon/Dropbox/ICML2021/data/objs --file_out eyeglasses.txt --scales 100

python -m mano_grasp.prepare_objects --help
python -m mano_grasp.generate_grasps --help
python -m mano_grasp.generate_grasps --models_file YCB_DATASET_objects.txt --path_out /home/dragon/Dropbox/ICML2021/data/demo
eyeglasses_0011_part_objs_none_motion_scale_100
whole_-0.5_-0.5_default
python -m mano_grasp.generate_grasps --models whole_-0.5_-0.5_scale_100 --path_out /home/dragon/Dropbox/ICML2021/data/demo -s 100 -g 10
python -m mano_grasp.generate_grasps --models models_025_mug_google_16k_model_watertight_5000def_scale_1000 --path_out /home/dragon/Dropbox/ICML2021/data/demo -s 100 -g 10
python -m mano_grasp.generate_grasps --models eyeglasses_0011_part_objs_none_motion_scale_100 --path_out /home/dragon/Dropbox/ICML2021/data/demo
python -m mano_grasp.generate_grasps --models phone glass --path_out /home/dragon/Dropbox/ICML2021/data/demo
python -m mano_grasp.generate_grasps --models eyeglasses_scale_100 --path_out /home/dragon/Dropbox/ICML2021/data/demo -s 200 -g 100
TODO: !!! animating a new articulated object
========== visualization with hand & object grasping ======
cd ../YCB_Affordance
conda activate manopth
# dict_keys(['body', 'link_in_contact', 'contacts', 'epsilon', 'pose', 'volume', 'pca_manorot', 'mano_trans', 'taxonomy', 'dofs', 'quality', 'pca_poses', 'mano_pose'])
#           ['body', 'link_in_contact', 'contacts', 'epsilon', 'pose', 'volume', 'mano_trans', 'dofs', 'quality', 'mano_pose']

#>>>>>>>>>>>>>>>>>> for data generation
# transform objs to a uniform mesh
cd ./haoi-pose/tools && python transform_and_save_meshs.py

# prepare objects for grasping
python -m mano_grasp.prepare_objects --models_folder /home/dragon/Documents/ICML2021/data/objs --file_out /home/dragon/Documents/ICML2021/data/eyeglasses.txt --scales 200

# call generate grasps n times
for i in 1 2 3 4 5
do
  echo "Looping ... number $i"
  python -m mano_grasp.generate_grasps --models_file /home/dragon/Documents/ICML2021/data/eyeglasses.txt --path_out /home/dragon/Documents/ICML2021/data/grasps
done
# python -m mano_grasp.generate_grasps --models whole_-0.5_-0.5_scale_100 --path_out /home/dragon/Dropbox/ICML2021/data/grasps

# save hand mesh & vertices, joints, contacts
python save_hand_mesh.py --viz
python evaluate_grasp.py --item=eyeglasses

# create URDF for hands
python create_hand_urdf.py

# Pybullet simulation
source activate py36

# render data
./blender -b --python /home/dragon/Dropbox/ICML2021/code/haoi-pose/tools/blender_render.py
./blender -b --python /home/dragon/ARCwork/6DPose2019/haoi-pose/tools/blender_render.py
./blender -b --python /home/lxiaol9/6DPose2019/haoi-pose/tools/blender_render.py


# >>>>>>>>>>>>>> preprocessing data
python preprocess_blender.py

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> for debug use <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
hao
cd dataset
python base.py
#
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/eyeglasses/1/viz/* /home/dragon/Documents/ICML2021/model/eyeglasses/1/viz/
