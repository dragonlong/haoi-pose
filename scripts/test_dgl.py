import dgl
import torch
# Source nodes for edges (2, 1), (3, 2), (4, 3)
src_ids = torch.tensor([2, 3, 4])
dst_ids = torch.tensor([1, 2, 3])
g = dgl.graph((src_ids, dst_ids))
