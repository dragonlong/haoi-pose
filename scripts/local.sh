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
