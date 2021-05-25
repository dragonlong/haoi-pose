# # 0.913r: # supervision in camera space, airplane, but add T estimation, use fps sampling,
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.913r' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True \
# eval=True save=True 2>&1 | tee evaluation/logs/0.913r.txt
#
# # 0.9141r: # corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.9141r' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=1 \
# eval=True save=True pre_compute_delta=True
# eval=True save=True 2>&1 | tee evaluation/logs/0.9141r.txt
#
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> car >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
# # 0.92r: # supervision in camera space, airplane, but add T estimation, use fps sampling,
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.92r' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True \
# eval=True save=True 2>&1 | tee evaluation/logs/0.92r.txt
#
# # 0.921r: # corrected, supervision in camera space, random R, car, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.921r' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=1 \
# eval=True save=True 2>&1 | tee evaluation/logs/0.921r.txt

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> chair <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
# 0.95r: # supervision in camera space, bottle, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.95r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True

python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.95r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.95r.txt

# 0.951r: # corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.951r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
eval=True save=True pre_compute_delta=True

python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.951r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
eval=True save=True 2>&1 | tee evaluation/logs/0.951r.txt

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> sofa >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
# 0.96r: # supervision in camera space,, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.96r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True

python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.96r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.96r.txt

# 0.961r: # corrected, supervision in camera space, random R,but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.961r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
eval=True save=True pre_compute_delta=True

# 0.961r: # corrected, supervision in camera space, random R,but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.961r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
eval=True save=True 2>&1 | tee evaluation/logs/0.961r.txt

# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BOTTLE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
# # 0.94r: # supervision in camera space, bottle, but add T estimation, use fps sampling,
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.94r' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True \
# eval=True save=True pre_compute_delta=True
# eval=True save=True 2>&1 | tee evaluation/logs/0.9141r.txt
#
# 0.941r: # corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.941r' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=1 \
# eval=True save=True pre_compute_delta=True
# eval=True save=True 2>&1 | tee evaluation/logs/0.9141r.txt
