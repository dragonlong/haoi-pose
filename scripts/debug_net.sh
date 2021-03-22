(Pdb) tr_agent.net.encoder.Gblock
ModuleList(
  (0): GSE3Res(
    (GMAB): ModuleDict(
      (v): GConvSE3Partial(structure=[(4, 0), (4, 1)])
      (k): GConvSE3Partial(structure=[(4, 0)])
      (q): G1x1SE3(structure=[(4, 0)])
      (attn): GMABSE3(n_heads=4, structure=[(4, 0), (4, 1)])
    )
    (project): G1x1SE3(structure=[(16, 0), (16, 1)])
    (add): GSum(structure=[(16, 0), (16, 1)])
  )
  (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
  (2): GSE3Res(
    (GMAB): ModuleDict(
      (v): GConvSE3Partial(structure=[(4, 0), (4, 1)])
      (k): GConvSE3Partial(structure=[(4, 0), (4, 1)])
      (q): G1x1SE3(structure=[(4, 0), (4, 1)])
      (attn): GMABSE3(n_heads=4, structure=[(4, 0), (4, 1)])
    )
    (project): G1x1SE3(structure=[(16, 0), (16, 1)])
    (add): GSum(structure=[(16, 0), (16, 1)])
  )
  (3): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
)


(Pdb) tr_agent.net.encoder.pre_modules
ModuleList(
  (0): GSE3Res(
    (GMAB): ModuleDict(
      (v): GConvSE3Partial(structure=[(4, 0), (4, 1)])
      (k): GConvSE3Partial(structure=[(4, 0)])
      (q): G1x1SE3(structure=[(4, 0)])
      (attn): GMABSE3(n_heads=4, structure=[(4, 0), (4, 1)])
    )
    (project): G1x1SE3(structure=[(16, 0), (16, 1)])
    (add): GSum(structure=[(16, 0), (16, 1)])
  )
  (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
)
(Pdb)
