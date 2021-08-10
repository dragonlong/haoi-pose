### R1
We are well aware of the reviewer’s concerns on intra-category shape variations and poor canonical shape reconstruction, and that is exactly where our novelty and main contribution come from--by proposing leveraging SE(3) equivariance and designing an effective framework. And our method doesn’t use residual pose predictions for progressive alignment, but could get accurate pose in a single forward pass during the inference stage. We don’t claim novelty over the branch design for canonical reconstruction and pose regression.

`For main contribution of the paper`:
1. We are the first one to explore self-supervised category-level 6D pose estimation from point clouds when there are no CAD models available, allowing intra-category shape variations, and occlusions-introduced partialness, no need of iterative alignment, and handling both symmetric and non-symmetric categories;
2. We propose the key idea of using networks preserving SE(3) equivariance, which could disentangle pose and shape effectively, and solve this problem elegantly;
3. Our proposed framework can achieve accurate pose estimation results that are comparable or even beat supervised SOTA methods, for both complete and partial point cloud input in a single forward pass. The secret is that by using rotational-invariant features, our model could learn to predict aligned shape reconstruction in canonical space for different instances through pure end-to-end self-supervised training, and by using rotation-equivariant features on divided SO(3) groups, our pose branch can give accurate SO(3) regression and associated translation;

`For ablation studies on effects of quality of reconstructed shape on pose estimation`, our proposed method actually aims at better disentanglement of shape and pose, so they could be well predicted but separately with either SE(3) invariant feature or SE(3) equivariant feature.

1. In our major experiments on both complete and partial input, we already show that our reconstructed canonical shapes are naturally aligned well against intra-category variations. Also for categories like symmetric bottle, the predicted pose is still quite accurate even the reconstructed shape is not complete(in poor quality), check further visualization in figure 3 and figure 4 in our supp.;
2. We have additional ablation studies on this in table 2 and figure 5 in our supplementary, which shows that if using general non-equivariant neural networks(like KPConv), we won’t be able to get category-level aligned canonical shape reconstructions, which would largely affect the accuracy of pose estimation, we also examined choices over equivariant network backbones for shape and corresponding pose estimations;
3. We further show ablation studies using different backbones for pose and shape separately, this design will have direct influence on the reconstructed shape quality, which we could check the influence on pose estimation;

- different backbones for pose and shape separately:
  
| Dataset           | Shape Backbone | Pose Backbone | Mean R_err | Median R_err | Mean T_err | Median T_err                        | 5degrees Acc. | 5degree0.05 |
| :---------------- | :------------- | :------------ | :--------- | :----------- | :--------- | :---------------------------------- | :------------ | :---------- |
| Complete airplane | EPN            | EPN           | 23.09      | 1.66         | /          | /                                   | 0.87          | /           |
| Complete airplane | EPN            | KPConv        | 21.24      | 1.70         | /          | /                                   | 0.88          | /           |
| Complete airplane | KPConv         | EPN           | 133.89     | 169.96       | /          | /                                   | 0.00          | /           |
| Partial airplane  | EPN            | EPN           | 3.31       | 1.46         | ?          | 0.02(from paper - sheet shows 0.04) | 0.95          | 0.76        |
| Partial airplane  | EPN            | KPConv        | 4.24       | 1.96         | 0.03       | 0.02                                | 0.93          | 0.85        |
| partial airplane  | KPConv         | EPN           | 134.34     | 174.98       | 0.07       | 0.08                                | 0.05          | 0.05        |

minor:
 For lines 65-68, we modified it as ‘the estimated 6D pose should naturally be equivariant with…;


### R2
1. We will properly address the reviewer’s suggestions over writing. Specifically, we will better organize paragraphs in the methods section into sub-sections, and better index our equations. Shorter sentences will be preferred. And we will also correct typos in both the main paper and our supplementary;
2. We tune contribution point 3 as ‘our proposed method achieves accurate pose estimation that are comparable or surpasses existing supervised SOTA methods’;
3. For ablation study over training data scale, we list the evaluation on the same validation set for chair and airplane, when:
   - 25% or 50% data by using less instances, but keep same number of viewpoints; 
   - 25% or 50% data by reducing viewpoints per instance, but keep same number of instance; 
   - 200% data by increasing the viewpoints; 

4. On table 1 for complete shapes, our results are actually comparable to EPN regarding the medium error. Regarding the mean errors, we observe that for certain instances, probably due to the weak symmetry of the input object and learned bias, the model would prefer the canonical shape to be upside down(car in supp. figure 5), or back to front(airplane in supp. figure 5), which causes 180-degrees flip, thus increasing the mean error. This is not a problem for the chair category. EPN is the SOTA method for pose estimation, however the weakness of EPN neural network is handling objects with symmetry, which has been well explained in the original paper. Due to its own limitation, EPN doesn’t perform well on symmetric objects like bottles compared to KPConv;
5. Compared to other equivariant networks like SE(3) transformer, EPN doesn’t need expensive spherical harmonics computation, and is both effective and relatively efficient;  
Quantitative ablations or examples of limitations to symmet, bias viewpoints distribution, or occlusions;

- ablation study over training data scale:
  
| Dataset           | Data scale      | Mean R_err | Median R_err | Mean T_err | Median T_err                        | 5degrees Acc. | 5degree0.05 |
| :---------------- | :-------------- | :--------- | :----------- | :--------- | :---------------------------------- | :------------ | :---------- |
| Complete airplane | 100% instance   | 23.09      | 1.66         | /          | /                                   | 0.87          | /           |
| Complete airplane | 25% instance    | 18.38      | 2.41         | /          | /                                   | 0.88          | /           |
| Complete airplane | 50% instance    | 20.22      | 2.38         | /          | /                                   | 0.86          | /           |
| Partial airplane  | 100% instance   | 3.31       | 1.46         | ?          | 0.02(from paper - sheet shows 0.04) | 0.95          | 0.76        |
| Partial airplane  | 25% instance    | 4.84       | 2.02         | 0.02       | 0.02                                | 0.94          | 0.87        |
| Partial airplane  | 50% instance    | 4.53       | 1.34         | 0.02       | 0.02                                | 0.94          | 0.87        |
| Partial airplane  | 25% viewpoints  | 3.00       | 1.49         | 0.02       | 0.02                                | 0.96          | 0.88        |
| Partial airplane  | 50% viewpoints  | 3.14       | 1.37         | 0.02       | 0.02                                | 0.95          | 0.88        |
| Partial airplane  | 200% viewpoints |            |              |            |                                     |               |             |


- Quantitative ablations or examples of limitations to symmetry, bias viewpoints distribution, or occlusions:

| Dataset           | view points   | Mean R_err | Median R_err | Mean T_err | Median T_err | 5degrees Acc. | 5degree0.05 |
| :---------------- | :------------ | :--------- | :----------- | :--------- | :----------- | :------------ | :---------- |
| Complete airplane | 100% instance | 23.09      | 1.66         | /          | /            | 0.87          | /           |

### R3
1. We will add reference to the mentioned related works to give a better backgrounds, but our paper is more focused on category-level 6D pose estimation with self-supervised methods, even though we do find the shape reconstruction results are impressive sometimes;
2. To analyze the impact of equivariant networks on the shape and pose separately, we add ablation studies on: 1. KPConv for pose + EPN for shape; 2. EPN for pose + KPConv for shape;
3. Here we provide experiments when we set the predefined rotation group to be with 20 elements, instead of using 60. This ‘multiple pose hypothesis’ is determined by our EPN backbone, since each pose prediction is assigned to a subspace of SO(3) space, only by using all of them we could cover the whole icosahedron rotation group and achieve rotational equivariance. The same MLP layer is shared to generate pose predictions per feature instead of using multiple heads. In our ablation study with SE(3) transformer backbone(which is also SE(3) equivariant), we directly regress poses instead of using multiple hypotheses, but the results are as good as our proposed method;
4. [TODO] The second car in Figure 2 looks strange, check car input visualization;
5. Our training process is usually stable for different categories with both complete and partial inputs, but not always the cases, here we show >=3 runs of the same experiments on car, airplane categories for complete point clouds and partial point clouds;

- different backbones for pose and shape separately:
  
| Dataset          | view points | Mean R_err | Median R_err | Mean T_err | Median T_err | 5degrees Acc. | 5degree0.05 |
| :--------------- | :---------- | :--------- | :----------- | :--------- | :----------- | :------------ | :---------- |
| Partial airplane | limited     |            |              |            |              |               |             |

- EPN20 VS EPN60:
  
| Dataset           | Backbone | Mean R_err | Median R_err | Mean T_err | Median T_err                        | 5degrees Acc. | 5degree0.05 |
| :---------------- | :------- | :--------- | :----------- | :--------- | :---------------------------------- | :------------ | :---------- |
| Complete airplane | EPN20    |            |              |            |                                     |               |             |
| Complete airplane | EPN60    | 23.09      | 1.66         | /          | /                                   | 0.87          | /           |
| Partial airplane  | EPN20    |            |              |            |                                     |               |             |
| Partial airplane  | EPN60    | 3.31       | 1.46         | ?          | 0.02(from paper - sheet shows 0.04) | 0.95          | 0.76        |

- repeated runs for training stability:
  
| Dataset           | run_id | Mean R_err       | Median R_err   | Mean T_err | Median T_err                                                 | 5degrees Acc.  | 5degree0.05 |
| :---------------- | :----- | :--------------- | :------------- | :--------- | :----------------------------------------------------------- | :------------- | :---------- |
| Complete airplane | 1,2,3  | 14.24,17.47,     | 1.53,1.26      | /          | /                                                            | 0.91,0.89      | /           |
| Complete car      | 1,2,3  | 15.46,14.16,9.95 | 2.19,1.74,1.94 | /          | /                                                            | 0.89,0.91,0.95 | /           |
| Partial airplane  | 1,2,3  | 3.91,3.47,       | 1.75,1.58,     | 0.02,0.02  | 0.02,0.02                                                    | 0.94,0.95,     | 0.86,0.88   |
| Partial car       | 1,2,3  | 138.19,136.51,   | 176.06,176.09  | 0.11,0.05  | 0.11,0.04(assuming "sdiff" in line 19 should be "tdiff_mid") | 0.18,0.17      | 0.00,0.13   |

