#!/bin/bash
#SBATCH -N 1
#SBATCH --exclude=ca235
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_normal_q
#SBATCH --account=CESCA-CV
#SBATCH -t 48:00:00
#SBATCH --mail-user lxiaol9@vt.edu
#SBATCH -J kpseg1
#SBATCH --output=history/%A_4.out
#SBATCH --error=history/%A_4.err
######################
# Begin work section: single frame, point input #
######################
# source ~/.merl_bashrc
which python3
EXP_NAME='pointmotion_2.5'
EXP_NUM='1'
MODEL_NAME='kpconv3d'
cd /home/lxiaol9/4DAutoSeg/${EXP_NAME}
python3 train_motion_kpconv.py exp_name=${EXP_NAME} exp_num=${EXP_NUM} model_name=${MODEL_NAME} \
		   first_subsampling_dl=0.1 first_features_dim=128 in_features_dim=2 n_frames=2 batch_num=1 in_radius=6 epoch_steps=500 checkpoint_gap=20 \
		   val_batch_num=1 validation_size=100 val_radius=8 max_gpu_points=32000 \
           GPU=1 2>&1 | tee ./outputs/train_${EXP_NAME}_${MODEL_NAME}_${EXP_NUM}.log

# Inference
python3 train_motion_kpconv.py exp_name=${EXP_NAME} exp_num=${EXP_NUM} model_name=${MODEL_NAME} \
		   n_frames=2 batch_num=1 in_radius=8 epoch_steps=40 checkpoint_gap=50 \
		   val_batch_num=1 validation_size=400 val_radius=8 \
           GPU=1 process_index=1 process_num=4 val_frame_skip=40 eval=True 2>&1 | tee ./outputs/inference_${EXP_NAME}_${MODEL_NAME}_${EXP_NUM}.log
