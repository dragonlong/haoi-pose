
# scp -r lxiaol9@newriver1.arc.vt.edu:/groups/CESCA-CV/external/ShapeNetCore.v2/02880940 .
# bottle,02876657,498
# mug, 03797390,214
# bowl,02880940,186
# can, 02946921,108
# jar,  03593526,596
# knife,03624134,424
# cellphone,02992529,831
# camera,02942699,113
# remote,04074963,66

EXP=2.06
CFG="generation.copy_input=True item='obman' name_dset='obman' exp_num='2.06' target_category='bottle' use_category_code=False"

# generate mesh in .off format
python generate_co.py ${CFG}

# visualize predicted mesh and reference mesh in html, online
cd tools
python viz_paired_mesh.py ${CFG}

# find all interested meshes and save them in one html file

# manually download Mesh & hand and visualize them in pair(with pyrender)

# eval miou, chamfer distance L1python generate_co.py generation.copy_input=True item='obman' name_dset='obman' exp_num=${EXP} use_noisy_nocs=False 2>&1 | tee outputs/occ_eval_${EXP}.log
# it finds all actual data points, then go over them in parallel
python eval_objs.py eval=True test.category_id=0 item='obman' name_dset='obman' exp_num=${EXP} 2>&1 | tee outputs/occ_eval_objs_${EXP}.log

# pose evaluation, compute GT
cd evaluation && python full_gt.py
# npcs baseline estimation
python baseline_npcs.py --item='eyeglasses' --domain='unseen'
# pose & relative
python eval_pose_err.py --item='eyeglasses' --domain='unseen' --nocs='ANCSH'
# 3d miou estimation
python compute_miou.py --item='eyeglasses' --domain='unseen' --nocs='ANCSH'
# add inter-penetration
