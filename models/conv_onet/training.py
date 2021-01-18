import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from common.d3_utils import (
    compute_iou, make_3d_grid, add_key,
)
from models.training import BaseTrainer
from common import bp
def breakpoint():
    import pdb;pdb.set_trace()

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.
    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples
    '''

    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False, cfg=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.cfg = cfg

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.
        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        if type(loss) is list:
            loss[0].backward()
            self.optimizer.step()
            return [loss[0], loss[1], loss[2]]
        else:
            loss.backward()
            self.optimizer.step()

            return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        batch_size = points.size(0)

        kwargs = {}

        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)
        if self.cfg.use_category_code:
            category_code = data['code'].to(device)
        else:
            category_code = None
        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, category_code,
                               sample=self.eval_sample, **kwargs)
        if type(p_out) is list:
            p_out = p_out[0]
        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,  category_code,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p   = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        c = self.model.encode_inputs(inputs)

        kwargs = {}
        # General points
        if self.cfg.use_category_code:
            s = data['code'].to(device)
        else:
            s = None
        # logits = self.model.decode(p, c, s, **kwargs).logits
        outputs = self.model.decode(p, c, s, **kwargs)
        if type(outputs) is list:
            logits = [output.logits for output in outputs]
        else:
            logits = outputs.logits
        if occ.size(-1) == 2:
            loss_o = F.binary_cross_entropy_with_logits(logits[0], occ[:, :, 0], reduction='none')
            loss_h = F.binary_cross_entropy_with_logits(logits[1], occ[:, :, 1], reduction='none')
            loss_i = loss_o + 0.1 * loss_h
        else:
            if self.cfg.model.num_occ_heads > 1: # by xiaolong
                logits = logits[np.arange(logits.shape[0]), :, data.get('category')] # [B, 2048, 8] --> [B, 2048], we only choose
            loss_i = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
        loss = loss_i.sum(-1).mean()
        if occ.size(-1) == 2:
            return [loss, loss_o.sum(-1).mean(), loss_h.sum(-1).mean()]
        else:
            return loss
