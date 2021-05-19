import pytorch3d
import torch
from pytorch3d.renderer import compositing as compositing
from pytorch3d.renderer.points import rasterize_points
from torch import nn as nn
import numpy as np

@torch.jit.script
def linear_alpha(dist_xy, radius: float):
    dists2 = dist_xy.permute(0, 3, 1, 2)
    dists_norm = dists2 / (radius * radius)
    alpha = 1 - dists_norm
    return alpha


@torch.jit.script
def exponential_alpha(dist_xy, radius: float):
    dists2 = dist_xy.permute(0, 3, 1, 2)
    dists_norm = dists2 / (radius * radius)
    dists_norm = dists_norm.clamp(min=0.0)
    alpha = torch.exp(-1 * dists_norm)
    return alpha

def bp():
    import pdb;pdb.set_trace()

class PointsRenderer(nn.Module):
    """
    Adapted from PyTorch3D's PointRenderer. A class for rendering a batch of points.
    The class combines rasterization, weight calculation, and compisiting
    """

    def __init__(self, render_cfg):
        super().__init__()

        # Rasterizing settings
        self.S = render_cfg.render_size
        self.K = render_cfg.points_per_pixel

        # Convert radius from pixels to NDC
        radius = render_cfg.radius
        self.r = 2 * radius / float(render_cfg.render_size)

        # Define weight computation
        if render_cfg.weight_calculation == "linear":
            self.calculate_weights = linear_alpha
        elif render_cfg.weight_calculation == "exponential":
            self.calculate_weights = exponential_alpha
        else:
            raise ValueError()

        # Define compositing
        if render_cfg.compositor == "alpha":
            self.compositor = compositing.alpha_composite
        elif render_cfg.compositor == "weighted_sum":
            self.compositor = compositing.weighted_sum
        elif render_cfg.compositor == "norm_weighted_sum":
            self.compositor = compositing.norm_weighted_sum
        else:
            raise ValueError()

    def forward(self, points, features) -> torch.Tensor:
        """
            points      FloatTensor     B x N x 3
            features    FloatTensor     B x N x F
        """
        # Rasterize points -- bins set heurisically
        pointcloud = pytorch3d.structures.Pointclouds(points, features=features)
        idx, zbuf, dist_xy = rasterize_points(pointcloud, self.S, self.r, self.K)
        bp()
        # Calculate PC coverage
        valid_pts = (idx >= 0).float()
        valid_ray = valid_pts[:, :, :, 0]

        # Calculate composite weights -- dist_xy is squared distance!!
        # Clamp weights to avoid 0 gradients or errors
        weights = self.calculate_weights(dist_xy, self.r)
        weights = weights.clamp(min=0.0, max=0.99)

        # Composite the raster for feats and depth
        idx = idx.long().permute(0, 3, 1, 2).contiguous()
        feats = pointcloud.features_packed().permute(1, 0)
        feats = self.compositor(idx, weights, feats)

        # == Rasterize depth ==
        # zero out weights -- currently applies norm_weighted sum
        w_normed = weights * (idx >= 0).float()
        w_normed = w_normed / w_normed.sum(dim=1, keepdim=True).clamp(min=1e-9)
        z_weighted = zbuf.permute(0, 3, 1, 2).contiguous() * w_normed.contiguous()
        z_weighted = z_weighted.sum(dim=1, keepdim=True)
        bp()
        return {
            "raster_output": {
                "idx": idx,
                "zbuf": zbuf,
                "dist_xy": dist_xy,
                "alphas": weights,
                "points": points,
                "feats": features,
            },
            "feats": feats,
            "depth": z_weighted,
            "mask": valid_ray,
            "valid_rays": valid_ray.mean(dim=(1, 2)),
            "valid_pts": valid_pts.mean(dim=(1, 2, 3)),
        }

class render_setting(object):
    def __init__(self):
        self.render_size = 128
        self.pointcloud_source = "other"
        self.radius = 2.0
        self.points_per_pixel = 16
        self.compositor = "norm_weighted_sum"
        self.weight_calculation = "exponential"

if __name__ == '__main__':
    render_cfg = render_setting()
    depth_render = PointsRenderer(render_cfg)
    names = ['test_46786_40_canon.txt', 'test_46786_24_canon.txt']
    points_batch = []
    for name in names:
        fn = f'/groups/CESCA-CV/ICML2021/model/modelnet40aligned/0.81/generation/{name}'
        points = np.loadtxt(fn)[:, :3].astype(np.float32)
        points_batch.append(points)
    points_tensor = torch.from_numpy(np.stack(points_batch, axis=0))
    feats_tensor = torch.ones_like(points_tensor)
    print(points_tensor.shape)
    data_dict = depth_render(points_tensor, feats_tensor)
