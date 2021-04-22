import numpy as np
import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F
from equivariant_attention.from_se3cnn.SO3 import rot as rot_mat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Optional, List, Union
from torch_geometric.typing import Adj, Size, Tensor

# work under float;
# add layerNorm,
# add out_dim,
# add residul connection,
# add multiple Channel for X

# no mapping of x, but only use addition with non-linearity

# helper functions
def bp():
    import pdb;pdb.set_trace()

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, xyz=None, k=20, idx=None, use_delta=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(xyz, k=k)   # (batch_size, num_points, k)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, numid_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, numid_dims)  -> (batch_size*num_points, numid_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)

    if use_delta:
        x_neighbours = x.view(batch_size*num_points, num_channels)[idx, :]
        x_neighbours = x_neighbours.view(batch_size, num_points, k, num_channels)
        x = x.view(batch_size, num_points, 1, num_channels).repeat(1, 1, k, 1)
        x_diff = x_neighbours - x
        x_diff_square = torch.sum(x_diff.pow(2),dim=-1, keepdim=True)

    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, numid_dims)
    x = x.view(batch_size, num_points, 1, numid_dims).repeat(1, 1, k, 1)

    if use_delta:
        feature = torch.cat((feature-x, x, ), dim=3).permute(0, 3, 1, 2).contiguous()
    else:
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()


    return feature      # (batch_size, 2*numid_dims, num_points, k)

def exists(val):
    return val is not None

def safe_div(num, den, eps = 1e-8):
    res = num.div(den.clamp(min = eps))
    res.masked_fill_(den == 0, 0.)
    return res

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

# swish activation fallback

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.fn = nn.Sequential(nn.LayerNorm(1), nn.GELU())

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        phase = self.fn(norm)
        return (phase * normed_coors)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats, coors, **kwargs):
        feats = self.norm(feats)
        feats, coors = self.fn(feats, coors, **kwargs)
        return feats, coors

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, feats, coors, **kwargs):
        feats_out, coors_delta = self.fn(feats, coors, **kwargs)
        return feats + feats_out, coors + coors_delta

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4 * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, feats, coors):
        return self.net(feats), 0

class EquivariantAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 4,
        edge_dim = 0,
        mid_dim = 16,
        fourier_features = 4,
        norm_rel_coors = False,
        num_nearest_neighbors = 0,
        only_sparse_neighbors = False,
        coor_attention = False,
        valid_neighbor_radius = float('inf'),
        init_eps = 1e-3
    ):
        super().__init__()
        self.fourier_features = fourier_features
        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_neighbor_radius = valid_neighbor_radius

        attn_inner_dim = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Linear(dim, attn_inner_dim * 3, bias = False)
        self.to_out = nn.Linear(attn_inner_dim, dim)

        pos_dim = (fourier_features * 2) + 1
        edge_input_dim = (dim_head * 2) + edge_dim

        self.to_pos_emb = nn.Sequential(
            nn.Linear(pos_dim, dim_head * 2),
            nn.ReLU(),
            nn.Linear(dim_head * 2, dim_head)
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_input_dim * 2, mid_dim),
            nn.ReLU()
        )

        self.to_attn_mlp = nn.Sequential(
            nn.Linear(mid_dim, mid_dim * 4),
            nn.ReLU(),
            nn.Linear(mid_dim * 4, 1),
            Rearrange('... () -> ...')
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(mid_dim, mid_dim * 4),
            nn.ReLU(),
            nn.Linear(mid_dim * 4, 1),
            Rearrange('... () -> ...')
        )

        self.rel_coors_norm = CoorsNorm() if norm_rel_coors else nn.Identity()

        self.coor_attention = coor_attention

        self.to_coors_out = nn.Sequential(
            nn.Linear(heads, 1, bias = False),
            Rearrange('... () -> ...')
        )

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(
        self,
        feats,
        coors,
        edges = None,
        mask = None,
        adj_mat = None
    ):
        b, n, d, h, fourier_features, num_nn, only_sparse_neighbors, valid_neighbor_radius, device = *feats.shape, self.heads, self.fourier_features, self.num_nearest_neighbors, self.only_sparse_neighbors, self.valid_neighbor_radius, feats.device

        assert not (only_sparse_neighbors and not exists(adj_mat)), 'adjacency matrix must be passed in if only_sparse_neighbors is turned on'

        if exists(mask):
            num_nodes = mask.sum(dim = -1)

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = rel_coors.norm(p = 2, dim = -1)

        # calculate neighborhood indices

        nbhd_indices = None
        nbhd_masks = None
        nbhd_ranking = rel_dist

        if exists(adj_mat):
            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat, 'i j -> b i j', b = b)

            self_mask = torch.eye(n, device = device).bool()
            self_mask = rearrange(self_mask, 'i j -> () i j')
            adj_mat.masked_fill_(self_mask, False)

            max_adj_neighbors = adj_mat.long().sum(dim = -1).max().item() + 1

            num_nn = max_adj_neighbors if only_sparse_neighbors else (num_nn + max_adj_neighbors)
            valid_neighbor_radius = 0 if only_sparse_neighbors else valid_neighbor_radius

            nbhd_ranking = nbhd_ranking.masked_fill(self_mask, -1.)
            nbhd_ranking = nbhd_ranking.masked_fill(adj_mat, 0.)

        if num_nn > 0:
            # make sure padding does not end up becoming neighbors
            if exists(mask):
                ranking_mask = mask[:, :, None] * mask[:, None, :]
                nbhd_ranking = nbhd_ranking.masked_fill(~ranking_mask, 1e5)

            nbhd_values, nbhd_indices = nbhd_ranking.topk(num_nn, dim = -1, largest = False)
            nbhd_masks = nbhd_values <= valid_neighbor_radius

        # calculate relative distance and optionally fourier encode

        rel_dist = rearrange(rel_dist, 'b i j -> b i j ()')

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        rel_dist = repeat(rel_dist, 'b i j d -> b h i j d', h = h)

        # derive queries keys and values

        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # calculate nearest neighbors

        i = j = n

        if exists(nbhd_indices):
            i, j = nbhd_indices.shape[-2:]
            nbhd_indices_with_heads = repeat(nbhd_indices, 'b n d -> b h n d', h = h)
            k         = batched_index_select(k, nbhd_indices_with_heads, dim = 2)
            v         = batched_index_select(v, nbhd_indices_with_heads, dim = 2)
            rel_dist  = batched_index_select(rel_dist, nbhd_indices_with_heads, dim = 3)
            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim = 2)
        else:
            k = repeat(k, 'b h j d -> b h n j d', n = n)
            v = repeat(v, 'b h j d -> b h n j d', n = n)

        rel_dist_pos_emb = self.to_pos_emb(rel_dist)

        # inject position into values

        v = v + rel_dist_pos_emb

        # prepare mask

        if exists(mask):
            q_mask = rearrange(mask, 'b i -> b () i ()')
            k_mask = repeat(mask, 'b j -> b i j', i = n)

            if exists(nbhd_indices):
                k_mask = batched_index_select(k_mask, nbhd_indices, dim = 2)

            k_mask = rearrange(k_mask, 'b i j -> b () i j')

            mask = q_mask * k_mask

            if exists(nbhd_masks):
                mask &= rearrange(nbhd_masks, 'b i j -> b () i j')

        # expand queries and keys for concatting

        q = repeat(q, 'b h i d -> b h i n d', n = j)

        edge_input = torch.cat(((q * k), rel_dist_pos_emb), dim = -1)

        if exists(edges):
            if exists(nbhd_indices):
                edges = batched_index_select(edges, nbhd_indices, dim = 2)

            edges = repeat(edges, 'b i j d -> b h i j d', h = h)
            edge_input = torch.cat((edge_input, edges), dim = -1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)

        if exists(mask):
            mask_value = -torch.finfo(coor_weights.dtype).max if self.coor_attention else 0.
            coor_weights.masked_fill_(~mask, mask_value)

        if self.coor_attention:
            coor_weights = coor_weights.softmax(dim = -1)

        rel_coors = self.rel_coors_norm(rel_coors)

        coors_out = einsum('b h i j, b i j c -> b i c h', coor_weights, rel_coors)
        coors_out = self.to_coors_out(coors_out)

        # derive attention

        sim = self.to_attn_mlp(m_ij)

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        # weighted sum of values and combine heads

        out = einsum('b h i j, b h i j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, coors_out

class EGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        mid_dim = 64,
        out_dim = 64,
        edge_dim = 0,
        layer_num = 1,
        x_channel = 1,
        fourier_features = 0,
        num_nearest_neighbors = 16,
        dropout = 0.0,
        init_eps = 1e-3,
        norm_feats = False,
        norm_coors = False,
        resi_connect = False,
        update_feats = True,
        update_coors = True,
        only_sparse_neighbors = False,
        valid_radius = float('inf'),
        m_pool_method = 'max',
        knn_method = 'max'
    ):
        super().__init__()
        assert m_pool_method in {'sum', 'mean', 'max'}, 'pool method must be either sum or mean'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'

        self.fourier_features = fourier_features

        edge_input_dim = (fourier_features * 2) + (in_dim * 2) + edge_dim + 1
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # here it is c --> 2*c --> mid_dim
        # 3-(6,64)->64-(128,64)->64-(128)->64 : concat 192-->1024, maxpooling & repeat: concat: 1024 + 192 --> 512 --> 256 --> 128 --> 3/6
        self.e_mlp = nn.Sequential(nn.Conv2d(edge_input_dim, mid_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(mid_dim),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(mid_dim),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.node_norm  = nn.LayerNorm(in_dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm() if norm_coors else nn.Identity()

        self.m_pool_method = m_pool_method
        self.knn_method    = knn_method
        self.resi_connect  = resi_connect

        self.f_mlp = nn.Sequential(
            nn.Linear(in_dim + mid_dim, mid_dim, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.LeakyReLU(negative_slope=0.2)
        ) if update_feats else None

        # self.x_mlp = nn.Sequential(nn.Conv2d(mid_dim, mid_dim * 4, kernel_size=1, bias=False),
        #                            nn.BatchNorm2d(mid_dim * 4),
        #                            nn.LeakyReLU(negative_slope=0.2),
        #                            nn.Conv2d(mid_dim * 4, x_channel, kernel_size=1, bias=False)) if update_coors else None
        self.x_mlp = nn.Sequential(
            nn.Linear(mid_dim, mid_dim * 4),
            dropout,
            SiLU(),
            nn.Linear(mid_dim * 4, x_channel)
        ) if update_coors else None

        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(self, feats, coors, edges = None, mask = None, adj_mat = None):
        b, n, d, device, fourier_features, num_nearest, valid_radius, only_sparse_neighbors = *feats.shape, feats.device, self.fourier_features, self.num_nearest_neighbors, self.valid_radius, self.only_sparse_neighbors
        use_nearest = num_nearest > 0 or only_sparse_neighbors
        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim = -1, keepdim = True)

        ranking = rel_dist[..., 0]
        if self.knn_method == 'max':
            nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim = -1)
        else:
            nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim = -1, largest=False)

        # nbhd_mask = nbhd_ranking <= valid_radius

        rel_coors = batched_index_select(rel_coors, nbhd_indices, dim = 2)
        rel_dist  = batched_index_select(rel_dist, nbhd_indices, dim = 2)

        if exists(edges):
            edges = batched_index_select(edges, nbhd_indices, dim = 2)
        # if fourier_features > 0:
        #     rel_dist = fourier_encode_dist(rel_dist, num_encodings = fourier_features)
        #     rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        feats_j = batched_index_select(feats, nbhd_indices, dim = 1)
        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i-feats_j, feats_j, rel_dist), dim = -1).permute(0, 3, 1, 2).contiguous()
        m_ij = self.e_mlp(edge_input).permute(0, 2, 3, 1).contiguous()

        if exists(self.x_mlp):
            coor_weights = self.x_mlp(m_ij)
            # coor_weights = self.x_mlp(m_ij.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous() # [B, N, k, cx]
            rel_coors = rearrange(rel_coors, 'b i j (m c) -> b i j m c', c=3)
            if exists(mask):
                coor_weights.masked_fill_(~mask, 0.)

            coors_out = einsum('b i j k, b i j m c -> b i k c', coor_weights, rel_coors) + rearrange(coors, 'b i (j k) -> b i j k', k=3)
            coors_out = rearrange(coors_out, 'b i k c -> b i (k c)')
        else:
            coors_out = coors

        if self.m_pool_method == 'mean':
            if exists(mask):
                mask_sum = m_ij_mask.sum(dim = -2)
                m_i = safe_div(m_ij.sum(dim = -2), mask_sum)
            else:
                m_i = m_ij.mean(dim = -2)

        elif self.m_pool_method == 'sum':
            m_i = m_ij.sum(dim = -2)

        elif self.m_pool_method == 'max': # original dgcnn is maxpooling
            m_i = m_ij.max(dim=-2, keepdim=False)[0]

        normed_feats = self.node_norm(feats)

        if exists(self.f_mlp):
            f_mlp_input = torch.cat((normed_feats, m_i), dim = -1) # need to concatenate feature & remap
            if self.resi_connect:
                node_out = self.f_mlp(f_mlp_input) + feats # use residul connection
            else:
                node_out = self.f_mlp(f_mlp_input)
        else:
            node_out = m_i

        return node_out, coors_out

class MLP(nn.Module):
    def __init__(self, dim, in_channel, mlp, use_bn=True, skip_last=True, last_acti=None):
        super(MLP, self).__init__()
        layers = []
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        bn = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
        last_channel = in_channel
        for i, out_channel in enumerate(mlp):
            layers.append(conv(last_channel, out_channel, 1))
            if use_bn and (not skip_last or i != len(mlp) - 1):
                layers.append(bn(out_channel))
            if (not skip_last or i != len(mlp) - 1):
                layers.append(nn.ReLU())
            last_channel = out_channel
        if last_acti is not None:
            if last_acti == 'softmax':
                layers.append(nn.Softmax(dim=1))
            elif last_acti == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                assert 0, f'Unsupported activation type {last_acti}'
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EquivariantDGCNN(nn.Module):
    def __init__(self, k=16, C_in=3, C_mid=64, emb_dims=1024, num_mode=1, depth=3, config=None):
        """
        k,
        C_mid,
        C_in=1,
        C_out=2,
        depth=4
        """
        # 3-(6,64)->64-(128,64)->64-(128)->64 : concat 192-->1024, maxpooling & repeat: concat: 1024 + 192 --> 512 --> 256 --> 128 --> 3/6
        super(EquivariantDGCNN, self).__init__()
        self.k = k # nsamples
        self.C_in  = C_in
        self.depth = depth
        self.is_equivalence = config.is_equivalence
        self.layers = nn.ModuleList([])
        for i in range(depth): # 4 blocks
            in_feat_dim  = C_mid
            mid_feat_dim = C_mid
            out_feat_dim = C_mid
            resi_connect = False
            if i == 0:
                in_feat_dim = C_in
                resi_connect = False
            self.layers.append(EGNN(in_dim = in_feat_dim, # 64
                    x_channel = 1,
                    mid_dim = mid_feat_dim,
                    out_dim = out_feat_dim,
                    num_nearest_neighbors = config.num_nearest_neighbors,
                    dropout = config.dropout,
                    resi_connect=config.resi_connect,
                    update_feats = config.update_feats,
                    update_coors = config.update_coors,
                    norm_coors = config.norm_coors,
                    norm_feats = config.norm_feats,
                    m_pool_method=config.m_pool_method,
                    knn_method = config.knn_method))

        # layer 4
        if self.is_equivalence:
            self.x_conv = nn.ModuleList([])
            for i in range(depth): # 4 blocks
                in_feat_dim  = C_mid
                mid_feat_dim = C_mid
                out_feat_dim = C_mid
                if i == 0:
                    in_feat_dim = C_mid * 3
                self.x_conv.append(EGNN(in_dim = in_feat_dim,
                        x_channel = 1,
                        mid_dim = mid_feat_dim,
                        out_dim = out_feat_dim,
                        num_nearest_neighbors = config.num_nearest_neighbors,
                        dropout = config.dropout,
                        resi_connect = config.resi_connect,
                        update_feats = config.update_feats,
                        update_coors = config.update_coors,
                        norm_coors = config.norm_coors,
                        norm_feats = config.norm_feats,
                        m_pool_method=config.m_pool_method,
                        knn_method = config.knn_method))

            self.heads_R = EGNN(in_dim = 64, x_channel = 2 * num_mode,
                    mid_dim = 3,
                    out_dim = 3,
                    num_nearest_neighbors = 16)

            self.heads_T = EGNN(in_dim = 64, x_channel = 1,
                    mid_dim = 3,
                    out_dim = 3,
                    num_nearest_neighbors = 16)

        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))

        # feature fusion: 1216 --> 512 --> 256 --> num_parts
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.15)

        head_cfg = {
            "R": [128, 6*num_mode, None],
            "T": [128, 3, None],
            "N": [128, 3, 'sigmoid'],
            "M": [128, num_mode, 'softmax'],
        }
        self.heads  = nn.ModuleDict()
        for key, mlp in head_cfg.items():
            self.heads[key] = MLP(dim=1, in_channel=256, mlp=mlp[:-1], use_bn=True, skip_last=True, last_acti=mlp[-1])

    def forward(self, pts=None, f=None, x=None, verbose=False):
        feats = []
        if pts is not None:
            x = pts[:, :3, :]
            if pts.shape[1] > 3:
                f = pts[:, 3:, :]
            else:
                f = x
            # standard input is [B, C, N]
            num_points = f.shape[-1]
        else:
            num_points = f.shape[1]

        if x.shape[1] == 3:
            x = x.permute(0, 2, 1).contiguous()
            f = f.permute(0, 2, 1).contiguous()
        # default input is [B, N, C]
        for i, layer in enumerate(self.layers):
            f, x = layer(f, x)
            feats.append(f.permute(0, 2, 1).contiguous())

        f = torch.cat(feats, dim=1)             # (batch_size, 64*3, num_points)
        if self.is_equivalence:
            f_x = f.permute(0, 2, 1).contiguous()
            for i, layer in enumerate(self.x_conv):
                f_x, x = layer(f_x, x)

        f = self.conv6(f)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        f = f.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        f = f.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        f = torch.cat([f] + feats, dim=1)       # (batch_size, 1024+64*3, num_points)

        f = self.conv7(f)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        f = self.conv8(f)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        f = self.dp1(f)

        pred_dict = {}
        # pred_dict.update({'R': output_R.permute(0, 2, 1).contiguous(), 'T': output_T.permute(0, 2, 1).contiguous()})
        for key, head in self.heads.items():
            pred_dict[key] = head(f)
        if self.is_equivalence:
            _, output_R  = self.heads_R(f_x, x)
            _, output_T  = self.heads_T(f_x, x)
            pred_dict.update({'R': output_R.permute(0, 2, 1).contiguous(), 'T': output_T.permute(0, 2, 1).contiguous()})
        return pred_dict

if __name__ == '__main__':
    BS = 12
    N  = 512
    C  = 3
    device = torch.device("cuda:0")
    # default_type = torch.DoubleTensor
    default_type = torch.FloatTensor
    torch.set_default_tensor_type(default_type)
    layer1 = EquivariantDGCNN(C_in=C).to(device)
    # layer1 = EGNN(dim = C, out_dim=C, x_channel = 2, resi_connect=True).to(device)
    for j in range(2):
        f = torch.randn(BS, N, C).to(device)
        x = torch.randn(BS, N, 3).to(device)
        tx, ty, tz = np.random.rand(1, 3)[0] * 180
        rot = rot_mat(tx, ty, tz).to(device) # 3 * 3
        print('rot:\n', rot)
        # rot = np.array([[0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, 0.5, 0], [0, 0, 1]])
        # rot = torch.from_numpy(rot).type(default_type).to(device)
        x_rotated = torch.matmul(x, rot)
        #
        pred1  = layer1(f=f, x=x)
        # f1, x1 = pred1['N'], pred1['R']
        # pred2 = layer1(f, x_rotated)
        # f2, x2 = pred2['N'], pred2['R']
        # x1_rotated = torch.matmul(x1.permute(0, 2, 1).contiguous().view(BS, N, -1, 3).contiguous(), rot)
        #  # (1, 16, 512), (1, 16, 3)
        # print(f1.shape, x1.shape)
        # x_diff_rotation = x2.permute(0, 2, 1).contiguous().view(BS, N, -1, 3).contiguous() - x1_rotated
        # f_diff_rotation = f2 - f1
        # print('\n')
        # print('x diff mean:', torch.mean(torch.abs(x_diff_rotation)))
        # print('x diff max:', torch.max(torch.abs(x_diff_rotation)))
        # print('f diff mean:', torch.mean(torch.abs(f_diff_rotation)))
        # print('f diff max:', torch.max(torch.abs(f_diff_rotation)))

# no downsampling

# no neighborhood fetching

# no multiple C mapping

# no multiple mode

#
