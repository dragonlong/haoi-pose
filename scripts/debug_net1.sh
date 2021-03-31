Final dataset size: 99029
using nocs_synthetic data  val
Final dataset size: 5418
EPOCH[1][0]:   0%|          | 0/49515 [00:00<?, ?it/s, nocs=1.81]train_aegan.py:235: RuntimeWarning: Mean of empty slice.
  wandb.log({f'test/{key}': np.array(value).mean(), 'step': clock.step})
/home/lxiaol9/anaconda3/envs/ptrans36/lib/python3.6/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
EPOCH[1][513]:   1%|          | 514/49515 [04:14<6:09:23,  2.21it/s, nocs=0.494]Traceback (most recent call last):
  File "train_aegan.py", line 197, in main
    tr_agent.train_func(data)
  File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/base.py", line 142, in train_func
    self.forward(data)
  File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/agent_ae.py", line 76, in forward
    self.predict_se3(data)
  File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/agent_ae.py", line 134, in predict_se3
    self.latent_vect = self.net.encoder(data['G'].to(device))
  File "/home/lxiaol9/anaconda3/envs/ptrans36/lib/python3.6/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/ae_gan/networks_ae.py", line 439, in forward
    h1, G1, r1, basis1 = self.down_modules[0](h0, Gin=G0, BS=self.batch_size) # 512-256
  File "/home/lxiaol9/anaconda3/envs/ptrans36/lib/python3.6/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/ae_gan/networks_ae.py", line 553, in forward
    Gmid, Gout, xyz_ind = self.down_g(Gin, BS=BS)
  File "/home/lxiaol9/anaconda3/envs/ptrans36/lib/python3.6/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/lxiaol9/3DGenNet2019/haoi-pose/models/ae_gan/networks_ae.py", line 155, in forward
    Gmid = dgl.batch(glist)
  File "/home/lxiaol9/anaconda3/envs/ptrans36/lib/python3.6/site-packages/dgl/batch.py", line 199, in batch
    ret_feat = _batch_feat_dicts(frames, ndata, 'nodes["{}"].data'.format(ntype))
  File "/home/lxiaol9/anaconda3/envs/ptrans36/lib/python3.6/site-packages/dgl/batch.py", line 237, in _batch_feat_dicts
    utils.check_all_same_schema(schemas, feat_dict_name)
  File "/home/lxiaol9/anaconda3/envs/ptrans36/lib/python3.6/site-packages/dgl/utils/checks.py", line 132, in check_all_same_schema
    name, i, schema, schemas[0]))
dgl._ffi.base.DGLError: Expect all graphs to have the same schema on nodes["_U"].data, but graph 1 got
        {'x': Scheme(shape=(3,), dtype=torch.float32), 'f': Scheme(shape=(3, 1), dtype=torch.float32)}
which is different from
        {'x': Scheme(shape=(3,), dtype=torch.float32), 'f': Scheme(shape=(3, 1), dtype=torch.float32), '_ID': Scheme(shape=(), dtype=torch.int64)}.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

wandb: Waiting for W&B process to finish, PID 297293
wandb: Program failed with code 1.  Press ctrl-c to abort syncing.
wandb:
wandb: Find user logs for this run at: /home/lxiaol9/3DGenNet2019/haoi-pose/wandb/run-20210329_212943-1z1efbig/logs/debug.log
wandb: Find internal logs for this run at: /home/lxiaol9/3DGenNet2019/haoi-pose/wandb/run-20210329_212943-1z1efbig/logs/debug-internal.log
