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
mkdir -p 0.813f/generation/
mkdir -p 0.813d/generation/
mkdir -p 0.813d1/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.813d/generation/test*txt 0.813d/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.813d1/generation/test*txt 0.813d1/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.813f/generation/test*txt 0.813f/generation/

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> complete shapes <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
mkdir -p 0.813/generation/
mkdir -p 0.851/generation/
mkdir -p 0.8581/generation/
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

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> partial data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.9141r/generation/test*txt 0.9141r/generation/

scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.94r/generation/test*txt 0.94r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.941r/generation/test*txt 0.941r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.95r/generation/test*txt 0.95r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.951r/generation/test*txt 0.951r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.96r/generation/test*txt 0.96r/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.961r/generation/test*txt 0.961r/generation/

cp -r 0.913r/ partial_airplane/
cp -r 0.921r/ partial_car/
cp -r 0.951r/ partial_chair/
cp -r 0.961r/ partial_sofa/
cp -r 0.941r/ partial_bottle/

zip -r partial_airplane.zip partial_airplane/
zip -r partial_bottle.zip partial_bottle/
zip -r partial_car.zip partial_car/
zip -r partial_chair.zip partial_chair/
zip -r partial_sofa.zip partial_sofa/

rclone sync . --include "{partial}*.{zip}" drive:Object_and_hands/results/preds/ -P

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ablation study <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
mkdir -p 0.913d/generation
mkdir -p 0.91f/generation # wrong double-chamfer
mkdir -p 0.91f1/generation # single mode, uni-chamfer
mkdir -p 0.911e/generation # double mode, uni-chamfer
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.91f/generation/test*txt 0.91f/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.91f1/generation/test*txt 0.91f1/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.911e/generation/test*txt 0.911e/generation/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40new/0.913d/generation/test*txt 0.913d/generation/

0.91f
0.91f1
0.911e # two modes
'canon': our predicted Z in canonical space
'target': reference NOCS
'input': input points with RT
'pred': transformed Z using RT predictions

h | g partial_airplane
 1903  cp -r 0.913r/ partial_airplane/
 1908  zip -r partial_airplane.zip partial_airplane/
 2005  h | g partial_airplane
h | g partial_car
 1899  cp -r 0.921r/ partial_car/
 1905  zip -r partial_car.zip partial_car/
 2004  h | g partial_car
 h | g partial_chair
  1901  cp -r 0.951r/ partial_chair/
  1906  zip -r partial_chair.zip partial_chair/
  2007  h | g partial_chair
h | g partial_sofa
 1902  cp -r 0.961r/ partial_sofa/
 1907  zip -r partial_sofa.zip partial_sofa/
 2006  h | g partial_sofa
h | g partial_bottle
 1900  cp -r 0.941r/ partial_bottle/
 1904  zip -r partial_bottle.zip partial_bottle/
 2008  h | g partial_bottle

 python viz_pyvista.py --target_entry partial_car
 [(0.5309164815493399, -3.769319975096356, 1.921953276803789),
  (0.5, 0.5, 0.5),
  (-0.8475796668394029, 0.16215458883588682, 0.5052866490219897)]
 python viz_pyvista.py --target_entry partial_sofa
 [(1.6142431664203394, 0.3717558586450132, 4.857982974530071),
  (0.5, 0.5, 0.5),
  (-0.9688074820127506, 0.0031137579585255124, 0.24779501065899134)]
 python viz_pyvista.py --target_entry partial_sofa
 [(0.5, 0.5, 5.0),
  (0.5, 0.5, 0.5),
  (-1.0, 0.0, 0.0)]
 python viz_pyvista.py --target_entry partial_sofa
 [(0.5, 0.5, 5.0),
  (0.5, 0.5, 0.5),
  (-1.0, 0.0, 0.0)]
 python viz_pyvista.py --target_entry partial_sofa
 [(-3.955601057111286, 3.0758286534868624, 2.277850097096413),
  (0.5, 0.5, 0.5),
  (0.06874823767700966, -0.48341465603108646, 0.872687773576926)]
 python viz_pyvista.py --target_entry partial_bottle
 [(2.5952189832029404, -4.452374507640515, 1.355610395273453),
  (0.5, 0.5, 0.5),
  (-0.060068642651314674, 0.14520779338842588, 0.9875760501902084)]
 python viz_pyvista.py --target_entry partial_airplane
 [(0.5, 0.5, 5.0),
  (0.5, 0.5, 0.5),
  (-1.0, 0.0, 0.0)]
 python viz_pyvista.py --target_entry partial_car
 [(3.4900215341750975, 3.4519476397528868, 2.111141321339517),
  (0.5, 0.5, 0.5),
  (-0.23706881250698866, -0.2693215875542848, 0.9334153741040299)]
 python viz_pyvista.py --target_entry partial_chair
 [(4.29269707970993, -2.0102320189096123, 3.493694886390554),
  (0.5, 0.5, 0.5),
  (-0.6587681047843262, -0.10724692248497691, 0.7446627973360616)]
 python viz_pyvista.py --target_entry partial_sofa
 [(-2.8929695341676167, 3.963088479451781, 2.97846745463794),
  (0.5, 0.5, 0.5),
  (0.25372528310130393, -0.38612518761463016, 0.8868657283967685)]
 python viz_pyvista.py --target_entry partial_bottle
 [(-4.799448973047865, -0.5000692310061894, 1.2508176311576813),
  (0.5, 0.5, 0.5),
  (0.13377648488659183, 0.0346694120842685, 0.9904049090938103)]

  #>>>>>>>>>>>>>>>>>>>>>>> abalation study
  mkdir -p 0.813d/generation/
  mkdir -p 0.813e/generation/
  mkdir -p 0.813f/generation/
  scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.813d/generation/test*txt 0.813d/generation/
  scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.813e/generation/test*txt 0.813e/generation/
  scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.813f/generation/test*txt 0.813f/generation/

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
