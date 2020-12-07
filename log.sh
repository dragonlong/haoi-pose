Day 52:
- 3d joints visualization, live viz;
- total visualization from Obman + FPHA;(pose)
- implicit functions over pose, articulation, contacts;


Day 51:
- we could estimate occupancy given observation data(image, partial point cloud, camera, noisy, voxels);
- the predictions could either be in 1*1*1 cannonical space, or camera space;
- voxelized local grid feature is much more better, but we could also have only a simple shape embedding;
- data matters with GT mesh shape or SDF, but it looks like we could leverage parametric models/synthetic data/ to create GT, or only pose is enough;
- learn category-level, view dependent, interaction or even articultaion-aware representations!!!!
In our pose estimation task, we only have this partial-view depth, which also has occlusion, why not jointly predict them?
so just take this partial-view depth, and generate features, to predict occupancy in cannonical space;
or use predicted NOCS to estimate? why not end-to-end? it should also works
You also have additional labels;
1. shape code/embedding;
2. directly use NOCS/partial pc to predict completion;
  - jittered NOCS() predict complete shape;
  - add pointnet++ features;
  -
But I need the sampled pts(3D, and other nearby query points!!! in canonical space) + GT occupancy labels, depth data + NOCS,
3.

Day 50:
use obman data for occupancy training;
https://contactpose.cc.gatech.edu/contactpose_explorer.html
https://github.com/xinghaochen/awesome-hand-pose-estimation#depth
occupancy-based training,
even with partial-view depth scan, we could still complete the full shape;

or is it possible to simply train for shape embedding?
or maybe it is better to add convolutional occu?

https://www.google.com/url?q=https://imperialcollegelondon.box.com/v/first-person-action-benchmark&sa=D&ust=1603064832799000&usg=AFQjCNGhiGI3DRyhkXcei-bW7Nb2txcSPw
fpaicvl
juice carton', 'milk bottle', 'salt' and 'liquid soap
https://github.com/guiggh/hand_pose_action

Day 40:
----------Now checking 3023: /groups/CESCA-CV/ICML2021/model/eyeglasses/0.94/preds/seen/0038_3_3_7.h5
>>>>>>>>>>>>>   seen    <<<<<<<<<<<<
 contacts_stat:
 contacts_err: [0.         0.02125977 0.0659022  ... 0.09021062 0.06462473 0.09849302]
 mean_contacts_err: 0.059820834547281265
 contacts_miou:0.9002290487739154
#
>>>>>>>>>>>>>   unseen    <<<<<<<<<<<<
 contacts_stat:
 contacts_err: [0.09441432 0.09657571 0.05720257 ... 0.03077268 0.07717242 0.05733614]
 mean_contacts_err: 0.059757016599178314
 contacts_miou:0.8932834311147076


Day 36: todo
- hands training with GT translation;
- contacts visualization;

Day 35:
well, take it easy!!!
- better visualization; hand skeleton--> Yes!
- contact points training loss;
  - objectness label & objectness score to see which vote cluster is good as center;
  - vote_xyz and vote_label;
  - center_label & aggregated_vote_xyz
-

Day 34:
- Model debugging with end-to-end MANO;(regression with R only);
- gt_check to see whether point cloud matches with mesh;


Day 32:
- flexible head from pose.yaml; Yes!
- data visualization, compare with original TF training; Yes!
- validation test; yes! try to overfit 100 examples

- loss! Yes!
- segmentation test; !

- h+o training & log;
print(pred_dict['partcls_per_point'][0, :, :2])

Day 31:
- new codebase for joint voting and regression in camera space;
  - loss for joints vectors, miou;
  -
  - dataset loader for msra data;
  - prediction io;
- PCA pose space for joints;
