cd ~/Documents/ICML2021/data/external/output/ShapeNet.build/026657$
scp *.tar* lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/external/output/ShapeNet.build/02876657/

bash dataset_shapenet/install.sh
/groups/CESCA-CV/external/output/ShapeNet/03797390/
img_choy2016  pointcloud.npz  points.npz

python train_occupancy.py config/pointcloud/onet.yaml 2>&1 | tee 1_train.log
#

python generate.py configs/pointcloud/onet_pretrained.yaml
