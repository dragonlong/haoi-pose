echo 0.91a: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.91a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.91a.txt

echo 0.91b: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.91b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.91b.txt

echo 0.92a: # random R, airplane, but add T estimation, dense per-point voting, R0, ca201 1
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.92a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.92a.txt

echo 0.92b: # random R, airplane, but add T estimation, dense per-point voting, R0, ca201 1
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.92b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.92b.txt

echo 0.95a: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.95a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.95a.txt

echo 0.95b: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.95b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.95b.txt

echo 0.96a: # n random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.96a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.96a.txt

echo 0.96b: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.96b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True 2>&1 | tee evaluation/logs/0.96b.txt
