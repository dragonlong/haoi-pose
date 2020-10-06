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
