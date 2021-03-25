ssh -p 8127 lxiaol9@128.173.88.229
ssh cc
yj
Hao // xl’s workspace
interact --partition=v100_dev_q --nodes=1 --ntasks-per-node=12 --gres=gpu:1 -A CESCA-CV


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
./blender -b --python /home/lxiaol9/6DPose2019/haoi-pose/tools/blender-renderer-cube.py

# >>>>>>>>>>>>>> preprocessing data
python preprocess_blender.py

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> for debug use <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
hao
cd dataset
python base.py
start=`date +%s`
python shapenet_samplepoints.py --start_idx=0 --group_by=800 2>&1 | tee shapenet_gt1.log &
python shapenet_samplepoints.py --start_idx=800 --group_by=1600 2>&1 | tee shapenet_gt2.log &
python shapenet_samplepoints.py --start_idx=1600 --group_by=2400 2>&1 | tee shapenet_gt3.log &
python shapenet_samplepoints.py --start_idx=2400 --group_by=4000
end=`date +%s`
runtime=$((end-start))
echo ${runtime}

hao
cd tools
for i in 0 1 2 3 4
do
python obman_samplepoints.py idx=$i num=10 &
done

hao
cd tools
for i in 5 6 7 8 9
do
python obman_samplepoints.py idx=$i num=10 &
done

#>>>>>>>>>> for evaluation
# viz
# scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/eyeglasses/1/viz/* /home/dragon/Documents/ICML2021/model/eyeglasses/1/viz/
# 1.46
EXP=1.48
item='obman'
domain='unseen'
mkdir -p /home/dragon/Documents/ICML2021/model/${item}/${EXP}/preds/${domain}/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/${item}/${EXP}/preds/${domain}/* /home/dragon/Documents/ICML2021/model/${item}/${EXP}/preds/${domain}/
mv ~/Downloads/0000*.h5 /home/dragon/Documents/ICML2021/model/${item}/${EXP}/preds/${domain}/

# cd /home/dragon/Documents/ICML2021/log
# EXP=1.35
# log
# 2.41 2.42
for EXP in 2.6 2.7 2.71
do
cd /home/dragon/Documents/ICML2021/new_log/ae_gan
item='obman'
mkdir ${EXP}
scp -r lxiaol9@newriver2.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/${item}/${EXP}/train.events/ ${EXP} &
scp -r lxiaol9@newriver2.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/${item}/${EXP}/val.events/ ${EXP} &
done

for EXP in 2.6 2.7 2.71
do
cd /home/dragon/Documents/ICML2021/new_log/ae_gan
item='obman'
mkdir ${EXP}
scp -r lxiaol9@newriver2.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/${item}/${EXP}/tb/ ${EXP} &
done

# for EXP in 1.42 1.43 1.44
# for EXP in 1.451 1.452 1.453 1.471 1.49
# for EXP in 2.07 2.071 2.072 2.073 2.3
for EXP in 2.31 2.32
do
cd /home/dragon/Documents/ICML2021/new_log
item='obman'
scp -r lxiaol9@newriver2.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/${item}/${EXP}/logs/ ${EXP}
done

cd /home/dragon/Documents/ICML2021/log
EXP=2.08
item='obman'
scp -r lxiaol9@newriver2.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/${item}/${EXP}/logs/ ${EXP}

# mesh training vis
EXP=2.03
item='obman'
domain='unseen'
mkdir -p /home/dragon/Documents/ICML2021/model/${item}/${EXP}/
scp -r lxiaol9@newriver2.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/obman/2.1/vis/ /home/dragon/Documents/ICML2021/model/${item}/${EXP}/

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> for evluation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
# mesh eval, 2.03, 2.07
EXP=2.07
item='obman'
domain='unseen'
mkdir -p /home/dragon/Documents/ICML2021/model/${item}/${EXP}/
scp -r lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/${item}/${EXP}/generation/ /home/dragon/Documents/ICML2021/model/${item}/${EXP}/


#
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/eyeglasses/1/viz/* /home/dragon/Documents/ICML2021/model/eyeglasses/1/viz/
mkdir -p /home/dragon/Documents/ICML2021/model/eyeglasses/1.0/preds/seen/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/eyeglasses/1.0/preds/seen/0001_0_* /home/dragon/Documents/ICML2021/model/eyeglasses/1.0/preds/seen/

cd /home/dragon/Documents/ICML2021/log
EXP=1.41
item='obman'
scp -r lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/obman/${EXP}/tb/ ${EXP}
# scp -rf lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/3DGenNet2019/haoi-pose/out/pointcloud/${EXP}/logs/* ${EXP}

scp -r lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/3DGenNet2019/haoi-pose/out/pointcloud/2.01/vis /home/dragon/Documents/ICML2021/results/val_pred/2.01/
# Press Enter, ~, . one after the other to disconnect from a frozen session.
#
for i in 0.1  0.2  0.3  0.4  0.5  3.9  3.91  5.9  5.91  8.12  8.14  pointnet2_charlesmsg_0  pointnet2_charlesmsg_1  pointnet2_charlesssg_1  pointnet2_meteornet_1
do
 echo $i
 mkdir -p $i/train
 mkdir -p $i/unseen
 mkdir -p $i/seen
done

mkdir -p /home/dragon/Documents/ICML2021/results/test_pred/obman/
for EXP_NUM in 2.4074 2.40941 2.4058 2.406971
do
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/results/test_pred/obman/${EXP_NUM}_unseen_part_rt_pn_general.npy /home/dragon/Documents/ICML2021/results/test_pred/obman/
done
# # scp lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/3DGenNet2019/haoi-pose/out/pointcloud/2.01/vis/800* /home/dragon/Documents/ICML2021/results/val_pred/2.01/
# scp -r lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/3DGenNet2019/haoi-pose/outputs/media /home/dragon/Dropbox/ICML2021/code/haoi-pose/outputs

# viz input occupancy points
python viz_helper.py

# obman sample
python obman_samplepoints.py
  # # how to get faster SDF value from mesh!!
  # 1. create watertight mesh with;
  #
  # 2. Sample points near the surface, and use libigl to get SDF value;
  #
  # 3. Use's occupancy's methods for decide whether points belong to volume or others;
  #
  # 4. Trimesh's code;
  #
  # 5. Use Pyrender to visualize the points;
# '04074963'
for category in '03797390' '02880940' '02946921' '03593526' '03624134' '02992529' '02942699'
do
zip -r ${category}.zip ./${category} &
done

rclone sync 03624134.zip drive:Object_and_hands/external/ShapeNetCore.v2/ -P
rclone sync . --include "*.{zip}" drive:Object_and_hands/external/ShapeNetCore.v2/ -P
pointnet++
2.4074
NOCS-se3
2.40941
rclone sync . --include "{2.40941,2.4058,2.406971}/checkpoints/*" drive:Object_and_hands/model/obman/ -P
2.4058(R, L2)
2.406971(T voting)

cd /home/dragon/Documents/ICML2021/results/preds
EXP=2.409196 # 2.40584 validation_9000*
mkdir ${EXP}
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/obman/${EXP}/generation/test* ./${EXP}
#

# video/gif
gifski --fps 24 -o output.gif *.png
ffmpeg -i img%02d.png -vf palettegen palette.png
ffmpeg -i img%02d.png -i palette.png -lavfi paletteuse video.gif
ffmpeg -i input.mp4 -an -vf “fps=15,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse” -f gif - | gifsicle -o output.gif
