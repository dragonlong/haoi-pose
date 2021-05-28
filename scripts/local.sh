# save hand mesh & vertices
conda activate manopth
python save_hand_mesh.py --viz

 # blender config env
python -m ensurepip --default-pip
./blender -b --python /home/dragon/Dropbox/ICML2021/code/haoi3d/tools/blender_render.py
./pip install imageio xml-python scipy
cd
python setup.py build_ext -i

filename = '/home/dragon/Dropbox/ICML2021/code/haoi3d/tools/blender_render.py'
exec(compile(open(filename).read(), filename, 'exec'))

filename = '/home/dragon/Dropbox/ICML2021/code/haoi3d/tools/blender.py'
exec(compile(open(filename).read(), filename, 'exec'))

python train_obman.py

# scp lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/3DGenNet2019/haoi-pose/out/pointcloud/2.01/vis/800* /home/dragon/Documents/ICML2021/results/val_pred/2.01/
scp -r lxiaol9@newriver1.arc.vt.edu:/home/lxiaol9/3DGenNet2019/haoi-pose/outputs/media /home/dragon/Dropbox/ICML2021/code/haoi-pose/outputs
# local
python viz_helper.py

rm outputs/media

scp -r lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/external/ShapeNetCore.v2/02880940/524eb99c8d45eb01664b3b9b23ddfcbc 02880940/

# 03797390 02876657 02880940 02942699 02946921 02992529 03593526 03797390 04074963
for i in 03624134
do
rsync -av --exclude='*manifold.obj' --exclude='*.npz' lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/external/ShapeNetCore.v2/$i /home/dragon/Documents/ICML2021/external/ShapeNetCore.v2/ &
done

#
mkdir -p 0.91/generation
mkdir -p 0.92/generation
mkdir -p 0.921/generation
mkdir -p 0.911/generation
mkdir -p 0.913r/generation
mkdir -p 0.9141r/generation
mkdir -p 0.94r/generation
mkdir -p 0.941r/generation
mkdir -p 0.95r/generation
mkdir -p 0.951r/generation
mkdir -p 0.96r/generation
mkdir -p 0.961r/generation
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.913r/generation/test*txt 0.913r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.9141r/generation/test*txt 9141r/generation/

scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.94r/generation/test*txt 0.94r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.941r/generation/test*txt 0.941r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.95r/generation/test*txt 0.95r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.951r/generation/test*txt 0.951r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.96r/generation/test*txt 0.96r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.961r/generation/test*txt 0.961r/generation/


cp -r 0.921r/ complete_car/
cp -r 0.941r/ complete_bottle/
cp -r 0.951r/ complete_chair/
cp -r 0.961r/ complete_sofa/
cp -r 0.913r/ complete_airplane/
zip -r complete_bottle.zip complete_bottle/
zip -r complete_car.zip complete_car/
zip -r complete_chair.zip complete_chair/
zip -r complete_sofa.zip complete_sofa/
zip -r complete_airplane.zip complete_airplane/
rclone sync . --include "{complete}*.{zip}" drive:Object_and_hands/results/preds/ -P

if self.target_category == 'airplane':
    self.exp_num    = '0.813'
elif self.target_category == 'car':
    self.exp_num    = '0.851'
elif self.target_category == 'chair':
    self.exp_num    = '0.8581'
elif self.target_category == 'sofa':
    self.exp_num    = '0.8591'
elif self.target_category == 'bottle':
    self.exp_num    = '0.8562'

mkdir -p 0.813/generation/
mkdir -p 0.851/generation/
mkdir -p 0.8518/generation/
mkdir -p 0.8591/generation/
mkdir -p 0.8562/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.851/generation/test*txt 0.851/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.813/generation/test*txt 0.813/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.8581/generation/test*txt 0.8581/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.8591/generation/test*txt 0.8591/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.8562/generation/test*txt 0.8562/generation/

cp -r 0.851/ complete_car/
cp -r 0.8562/ complete_bottle/
cp -r 0.8581/ complete_chair/
cp -r 0.8591/ complete_sofa/
cp -r 0.813/ complete_airplane/
zip -r complete_bottle.zip complete_bottle/
zip -r complete_car.zip complete_car/
zip -r complete_chair.zip complete_chair/
zip -r complete_sofa.zip complete_sofa/
zip -r complete_airplane.zip complete_airplane/


'canon': our predicted Z in canonical space
'target': reference NOCS
'input': input points with RT
'pred': transformed Z using RT predictions
