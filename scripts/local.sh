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

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> for evluation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
#
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/eyeglasses/1/viz/* /home/dragon/Documents/ICML2021/model/eyeglasses/1/viz/
mkdir -p /home/dragon/Documents/ICML2021/model/eyeglasses/1.0/preds/seen/
scp lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/ICML2021/model/eyeglasses/1.0/preds/seen/0001_0_* /home/dragon/Documents/ICML2021/model/eyeglasses/1.0/preds/seen/
