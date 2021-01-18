import torch
import torch.nn as nn
from torch import distributions as dist
from models.conv_onet.model import decoder
from common import bp
# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}


class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''
    def __init__(self, decoder, encoder=None, device=None, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.decoder = decoder.to(device)
        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, p, inputs, s=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, s, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, s=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        logits = self.decoder(p, c, s, **kwargs)
        if type(logits) is list:
            p_rs = []
            for logit in logits:
                p_r = dist.Bernoulli(logits=logit)
                p_rs.append(p_r)
            return p_rs
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
