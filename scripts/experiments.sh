export PATH="/home/lxiaol9/anaconda3/bin:$PATH"
cd
. scripts/ai_power1.sh
module load cuda/10.0.130
source activate merl

screen -X -S  quit

0.1: # same model, but with new loss
  python train.py verbose=True gpu=0 name_model='pointnet2_meteornet' exp_num='0.1' TRAIN.loss='xentropy'

0.2: # new model, new loss
  python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.2' TRAIN.loss='xentropy' MODEL.arch_decoder='kaolin' 2>&1 | tee train_0.2.log

0.3: # new mode, but with same loss, ca197
  python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.3' TRAIN.loss='miou' MODEL.arch_decoder='kaolin' 2>&1 | tee train_0.3.log
  0.31: # with full unitvec loss
  python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.31' TRAIN.loss='miou' MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_0.31.log
  python post_hand.py --hand --exp_num=0.31 2>&1 | tee eval_0.31.log &
  python post_hand.py --hand --exp_num=0.31 2>&1 | tee eval_0.31_regression.log &
  python post_hand.py --hand --exp_num=0.31 --domain='unseen' 2>&1 | tee eval_0.31_regression.log &
  0.32: # with normal unitvec loss
  python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.32' TRAIN.loss='miou' MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_0.32.log
  python post_hand.py --hand --exp_num=0.32 2>&1 | tee eval_0.32.log
  python post_hand.py --hand --exp_num=0.32 2>&1 | tee eval_0.32_regression.log &
  python post_hand.py --hand --exp_num=0.32 --domain='unseen' 2>&1 | tee eval_0.32_unseen_regression.log &

  0.33: # with normal unitvec loss, with svd rotation and mean
  python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.33' TRAIN.loss='miou' MODEL.arch_decoder='kaolin' rot_align=True 2>&1 | tee outputs/train_0.33.log
  python post_hand.py --hand --exp_num=0.33 2>&1 | tee eval_0.33.log &
  python post_hand.py --hand --exp_num=0.33 2>&1 | tee eval_0.33_regression.log &
  python post_hand.py --hand --exp_num=0.33 --domain='unseen' 2>&1 | tee eval_0.33_unseen_regression.log &

0.4: # mano regression, 61 params, shape 10, articulation 45, global 6, regression is easy: but recontruction loss + mano_layer
  python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.4' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.4.log
  python post_hand.py --hand --exp_num=0.4 --mano 2>&1 | tee eval_0.4_mano.log &

# 0.5: # mano regression, 61 params, shape 10, articulation 45, global 6, regression is easy: but recontruction loss + mano_layer
#   python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.5' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.5.log
#   python post_hand.py --hand --exp_num=0.5 --mano 2>&1 | tee eval_0.5_mano.log &
#
# 0.6: only train with regressionR, regressionPose, partcls loss
#   python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.6' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.6.log
#   python post_hand.py --hand --exp_num=0.6 --mano 2>&1 | tee eval_0.6_mano.log &

0.7: # only train with regressionR, regressionPose, partcls loss, NOCS loss--> 6D vector directly,
  python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.7' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.7.log
  python post_hand.py --hand --exp_num=0.7 --mano 2>&1 | tee eval_0.7_mano.log &

0.8: # only train with regressionR, regressionPose, partcls loss, NOCS loss--> 6D vector directly, add vertices, joints loss
  python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.8' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.8.log
  python post_hand.py --hand --exp_num=0.8 --mano 2>&1 | tee eval_0.8_mano.log &
  python pred_check.py --exp_num=0.8 --hand --mano


0.9: # retrain with larger learning rate, 0.1-0.3
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.9' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.9.log
python post_hand.py --hand --exp_num=0.9 --mano 2>&1 | tee eval_0.8_mano.log &

0.91: # train with normal setting, 0.3, 0.6
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.91' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.91.log

0.92: # train with normal setting, 0.2, 0.4
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.92' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.92.log

0.93: # train with normal setting, 0.2, 0.3
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.93' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.93.log

0.94: # retrain with larger learning rate, 0.1-0.2, add confidence prediction after 1st epoch
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.94' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.94.log
#
0.95: # retrain with larger learning rate, 0.1-0.2, add 1 * center_loss after 1st epoch
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.95' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.95.log

# 1.0:
#   python train.py verbose=True gpu=1 name_model='pointnet2_meteornet' exp_num=1
#   python train.py verbose=True gpu=1 name_model='pointnet2_meteornet' exp_num=1 eval=True
#
# 1.1:
#   python train.py verbose=True gpu=1 name_model='pointnet2_charlesssg' ca218
