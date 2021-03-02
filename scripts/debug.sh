# 2.40971, 2.40972, 2.40973
EPOCH[1][7700]:  46%|██████████████████████████▍                              | 7700/16639 [1:56:04<2:14:44,  1.11it/s, regressionR=2e-7, seg=0.729, confidence=0.0322]
Traceback (most recent call last):
  File "train_aegan.py", line 224, in main
    tr_agent.val_func(data)
  File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/base.py", line 147, in val_func
    self.forward(data)
  File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/agent_ae.py", line 111, in forward
    self.confidence_good = pred_c[dis<THRESHOLD_GOOD].mean()
IndexError: too many indices for tensor of dimension 1

# 2.40911
h2, G2, r2, basis2 = self.down_modules[1](h1, Gin=G1) # 256-128
File "/home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages/torch/nn/modules/module.py", line 532, in __call__
result = self.forward(*input, **kwargs)
File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/ae_gan/networks_ae.py", line 555, in forward
Gmid, Gout, xyz_ind = self.down_g(Gin)
File "/home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages/torch/nn/modules/module.py", line 532, in __call__
result = self.forward(*input, **kwargs)
File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/ae_gan/networks_ae.py", line 147, in forward
g.edata['d'] = pos[i][dst] - pos[i][src]
IndexError: tensors used as indices must be long, byte or bool tensors


SE31x1, GSum


(Pdb) print(z.keys())
dict_keys(['0', '1'])
(Pdb) print(z['0'].shape, z['1'].shape)
torch.Size([1024, 4, 1]) torch.Size([1024, 4, 3])
(Pdb) n
> /home/lxiaol9/3DGenNet2019/se3-transformer-public/equivariant_attention/modules.py(618)forward()
-> z = self.add(z, features)
(Pdb) print(z['0'].shape, z['1'].shape)
torch.Size([1024, 16, 1]) torch.Size([1024, 16, 3])
(Pdb) print(features['0'].shape, features['1'].shape)
*** KeyError: '1'
(Pdb) print(features['0'].shape)
torch.Size([1024, 1, 1])
(Pdb) z = self.add(z, features)
(Pdb) print(z['0'].shape, z['1'].shape)
torch.Size([1024, 16, 1]) torch.Size([1024, 16, 3])
