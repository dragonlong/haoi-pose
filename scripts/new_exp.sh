# 2.4058 multiple instance, dynamic graph, with augmentation, L2
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4058' DATASET.train_batch=2 DATASET.test_batch=2 \
use_objective_R=True augment=True rotation_loss_type=1 eval=True save=True
use_wandb=True

# 2.406971, multiple instance, dynamic graph, with augmentation, T voting
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.406971' DATASET.train_batch=2 DATASET.test_batch=2 \
use_objective_T=True augment=True eval=True save=True
use_wandb=True

---> 2.40941: # use type 0 predict NOCS on complete shape, NOCS
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40941' DATASET.train_batch=2 DATASET.test_batch=2 \
augment=True pred_nocs=True use_objective_N=True \
eval=True save=True
use_wandb=True

2.4074: # pointnet++ on complete data for NOCS
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4074' DATASET.train_batch=2 DATASET.test_batch=2 \
augment=True pred_nocs=True use_objective_N=True \
eval=True save=True

use_wandb=True

2.41:
  2.411 # jar, AE-Graph, synthetic complete pcloud
  TRAIN_OBJ='python train_aegan.py training=ae_gan use_wandb=True vis=True n_pts=256 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ target_category='jar' exp_num='2.411'  DATASET.train_batch=2 DATASET.test_batch=2

  2.412: # jar,
  TRAIN_OBJ='python train_aegan.py training=ae_gan use_wandb=True vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ target_category='jar' exp_num='2.402' DATASET.train_batch=4 DATASET.test_batch=4

  2.413: # jar
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='jar' exp_num='2.413' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True use_wandb=True

  2.414: # multiple instance, dynamic graph, with augmentation, L2, jar
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='jar' exp_num='2.414' DATASET.train_batch=2 DATASET.test_batch=2 \
use_objective_R=True augment=True rotation_loss_type=1 \
use_wandb=True
  eval=True save=True

  2.415: # multiple instance, dynamic graph, with augmentation, T voting, jar
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='jar' exp_num='2.415' DATASET.train_batch=2 DATASET.test_batch=2 \
use_objective_T=True augment=True \
use_wandb=True
   eval=True save=True

  2.416: # use type 0 predict NOCS on complete shape, NOCS, jar
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='jar' exp_num='2.416' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True pred_nocs=True use_objective_N=True \
  use_wandb=True
  eval=True save=True


2.42 # all categories, AE, synthetic complete pcloud
  python train_aegan.py training=ae_gan name_model=ae dataset_class='HandDatasetAEGan' exp_num='2.42'


2.43: remote
  2.431: # multiple instance, dynamic graph, with augmentation, L2, remote
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='remote' exp_num='2.431' DATASET.train_batch=2 DATASET.test_batch=2 \
use_objective_R=True augment=True rotation_loss_type=1 \
use_wandb=True
  eval=True save=True

  2.432: # multiple instance, dynamic graph, with augmentation, T voting, remote
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='remote' exp_num='2.432' DATASET.train_batch=2 DATASET.test_batch=2 \
use_objective_T=True augment=True \
use_wandb=True
  eval=True save=True

  2.433: # use type 0 predict NOCS on complete shape, NOCS, remote
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='remote' exp_num='2.433' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True pred_nocs=True use_objective_N=True \
  use_wandb=True
  eval=True save=True


2.44: cellphone

2.45: camera

  2.451: # multiple instance, dynamic graph, with augmentation, L2, camera
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='camera' exp_num='2.451' DATASET.train_batch=2 DATASET.test_batch=2 \
use_objective_R=True augment=True rotation_loss_type=1 \
use_wandb=True
   eval=True save=True

  2.452: # 2.406971, multiple instance, dynamic graph, with augmentation, T voting, camera
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='camera' exp_num='2.452' DATASET.train_batch=2 DATASET.test_batch=2 \
use_objective_T=True augment=True \
use_wandb=True
eval=True save=True

  2.453: # use type 0 predict NOCS on complete shape, NOCS, camera
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='camera' exp_num='2.453' DATASET.train_batch=2 DATASET.test_batch=2 \
augment=True pred_nocs=True use_objective_N=True \
use_wandb=True
eval=True save=True

2.46: knife
