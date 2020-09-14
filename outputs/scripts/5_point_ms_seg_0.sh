#!/bin/bash
#SBATCH -N 1
#SBATCH --exclude=ca235
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_normal_q
#SBATCH --account=CESCA-CV
#SBATCH -t 48:00:00
#SBATCH --mail-user lxiaol9@vt.edu
#SBATCH -J seg0
#SBATCH --output=history/%A_4.out
#SBATCH --error=history/%A_4.err
######################
# Begin work section: single frame, point input #
######################
# source ~/.merl_bashrc
which python3

EXP_NUM='0'
EXP_NAME='pointseg_beta'
MODEL_NAME='kpconv'

cd /home/lxiaol9/4DAutoSeg/${EXP_NAME}
python3 train_point_kpconv.py exp_name=${EXP_NAME} exp_num=${EXP_NUM} model_name=${MODEL_NAME} \
		   num_classes=26 n_frames=2 batch_num=1 TRAIN.batch_size_per_gpu=4 n_frames=2 in_radius=4 epoch_steps=12000 checkpoint_gap=20 \
		   val_batch_num=1 validation_size=300 val_radius=8 input_threads=8 \
    	   2>&1 | tee ./outputs/train_${EXP_NAME}_${MODEL_NAME}_${EXP_NUM}.log

