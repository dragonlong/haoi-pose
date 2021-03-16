-----pointAEPose architecture-----
PointAE(
  (encoder): SE3Transformer(
    (pre_modules): ModuleList(
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
    (down_modules): ModuleList(
      (0): SE3TBlock(
        (down_g): InterDownGraph(
          (n_sampler): Sample()
          (e_sampler): SampleNeighbors()
        )
        (stage1): ModuleList(
          (0): GSE3Res(
            (GMAB): ModuleDict(
              (v): GConvSE3Partial(structure=[(4, 0), (4, 1)])
              (k): GConvSE3Partial(structure=[(4, 0), (4, 1)])
              (q): G1x1SE3(structure=[(4, 0), (4, 1)])
              (attn): GMABSE3(n_heads=4, structure=[(4, 0), (4, 1)])
            )
            (project): G1x1SE3(structure=[(16, 0), (16, 1)])
            (add): GSum(structure=[(16, 0), (16, 1)])
          )
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
        (stage2): ModuleList(
          (0): GSE3Res(
            (GMAB): ModuleDict(
              (v): GConvSE3Partial(structure=[(4, 0), (4, 1)])
              (k): GConvSE3Partial(structure=[(4, 0), (4, 1)])
              (q): G1x1SE3(structure=[(4, 0), (4, 1)])
              (attn): GMABSE3(n_heads=4, structure=[(4, 0), (4, 1)])
            )
            (project): G1x1SE3(structure=[(16, 0), (16, 1)])
            (add): GSum(structure=[(16, 0), (16, 1)])
          )
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
      )
      (1): SE3TBlock(
        (down_g): InterDownGraph(
          (n_sampler): Sample()
          (e_sampler): SampleNeighbors()
        )
        (stage1): ModuleList(
          (0): GSE3Res(
            (GMAB): ModuleDict(
              (v): GConvSE3Partial(structure=[(8, 0), (8, 1)])
              (k): GConvSE3Partial(structure=[(8, 0), (8, 1)])
              (q): G1x1SE3(structure=[(8, 0), (8, 1)])
              (attn): GMABSE3(n_heads=4, structure=[(8, 0), (8, 1)])
            )
            (project): G1x1SE3(structure=[(32, 0), (32, 1)])
            (add): GSum(structure=[(32, 0), (32, 1)])
          )
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
        (stage2): ModuleList(
          (0): GSE3Res(
            (GMAB): ModuleDict(
              (v): GConvSE3Partial(structure=[(8, 0), (8, 1)])
              (k): GConvSE3Partial(structure=[(8, 0), (8, 1)])
              (q): G1x1SE3(structure=[(8, 0), (8, 1)])
              (attn): GMABSE3(n_heads=4, structure=[(8, 0), (8, 1)])
            )
            (project): G1x1SE3(structure=[(32, 0), (32, 1)])
            (add): GSum(structure=[(32, 0), (32, 1)])
          )
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
      )
      (2): SE3TBlock(
        (down_g): InterDownGraph(
          (n_sampler): Sample()
          (e_sampler): SampleNeighbors()
        )
        (stage1): ModuleList(
          (0): GSE3Res(
            (GMAB): ModuleDict(
              (v): GConvSE3Partial(structure=[(16, 0), (16, 1)])
              (k): GConvSE3Partial(structure=[(16, 0), (16, 1)])
              (q): G1x1SE3(structure=[(16, 0), (16, 1)])
              (attn): GMABSE3(n_heads=4, structure=[(16, 0), (16, 1)])
            )
            (project): G1x1SE3(structure=[(64, 0), (64, 1)])
            (add): GSum(structure=[(64, 0), (64, 1)])
          )
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
        (stage2): ModuleList(
          (0): GSE3Res(
            (GMAB): ModuleDict(
              (v): GConvSE3Partial(structure=[(16, 0), (16, 1)])
              (k): GConvSE3Partial(structure=[(16, 0), (16, 1)])
              (q): G1x1SE3(structure=[(16, 0), (16, 1)])
              (attn): GMABSE3(n_heads=4, structure=[(16, 0), (16, 1)])
            )
            (project): G1x1SE3(structure=[(64, 0), (64, 1)])
            (add): GSum(structure=[(64, 0), (64, 1)])
          )
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
      )
      (3): SE3TBlock(
        (down_g): InterDownGraph(
          (n_sampler): Sample()
          (e_sampler): SampleNeighbors()
        )
        (stage1): ModuleList(
          (0): GSE3Res(
            (GMAB): ModuleDict(
              (v): GConvSE3Partial(structure=[(16, 0), (16, 1)])
              (k): GConvSE3Partial(structure=[(16, 0), (16, 1)])
              (q): G1x1SE3(structure=[(16, 0), (16, 1)])
              (attn): GMABSE3(n_heads=4, structure=[(16, 0), (16, 1)])
            )
            (project): G1x1SE3(structure=[(64, 0), (64, 1)])
            (add): GSum(structure=[(64, 0), (64, 1)])
          )
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
        (stage2): ModuleList(
          (0): GSE3Res(
            (GMAB): ModuleDict(
              (v): GConvSE3Partial(structure=[(16, 0), (16, 1)])
              (k): GConvSE3Partial(structure=[(16, 0), (16, 1)])
              (q): G1x1SE3(structure=[(16, 0), (16, 1)])
              (attn): GMABSE3(n_heads=4, structure=[(16, 0), (16, 1)])
            )
            (project): G1x1SE3(structure=[(64, 0), (64, 1)])
            (add): GSum(structure=[(64, 0), (64, 1)])
          )
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
      )
    )
    (up_modules): ModuleList(
      (0): GraphFPModule(
        (Tblock): ModuleList(
          (0): GConvSE3(structure=[(64, 0), (64, 1)], self_interaction=True)
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
      )
      (1): GraphFPModule(
        (Tblock): ModuleList(
          (0): GConvSE3(structure=[(32, 0), (32, 1)], self_interaction=True)
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
      )
      (2): GraphFPModule(
        (Tblock): ModuleList(
          (0): GConvSE3(structure=[(32, 0), (32, 1)], self_interaction=True)
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
      )
      (3): GraphFPModule(
        (Tblock): ModuleList(
          (0): GConvSE3(structure=[(32, 0), (32, 1)], self_interaction=True)
          (1): GNormSE3(num_layers=0, nonlin=ReLU(inplace=True))
        )
      )
    )
    (Oblock): ModuleList(
      (0): GConvSE3(structure=[(128, 0)], self_interaction=True)
      (1): GConvSE3(structure=[(4, 0), (4, 1)], self_interaction=True)
      (2): GConvSE3(structure=[(1, 0), (1, 1)], self_interaction=True)
    )
    (Pblock): GAvgPooling(
      (pool): AvgPooling()
    )
  )
  (regressor): RegressorFC()
  (classifier_mode): RegressorC1D(
    (module): Sequential(
      (0): Conv1d(4, 2, kernel_size=(1,), stride=(1,))
      (1): LeakyReLU(negative_slope=0.01, inplace=True)
      (2): Softmax(dim=1)
    )
  )
  (decoder): DecoderFC(
    (model): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): LeakyReLU(negative_slope=0.01, inplace=True)
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): LeakyReLU(negative_slope=0.01, inplace=True)
      (4): Linear(in_features=256, out_features=1536, bias=True)
    )
  )
)
