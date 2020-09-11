#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020

import numpy as np
import __init__
from modules.conv_layer import *

def breakpoint():
    import pdb;pdb.set_trace()

def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))


            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim


            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        breakpoint()
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total

class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.task = config.dataset_task
        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)

        # add motion
        self.mhead_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.mhead_softmax = UnaryBlock(config.first_features_dim, 3, False, 0) # motion
        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # add loss for motion
        motion_w = torch.from_numpy(np.array([0, 1, 5], dtype=np.float32))
        self.criterion_m = torch.nn.CrossEntropyLoss(weight=motion_w, ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)


        m = self.mhead_mlp(x, batch)
        m = self.mhead_softmax(m, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)


        return x, m

    # def loss(self, outputs, labels):
    #     """
    #     Runs the loss on outputs of the model
    #     :param outputs: logits
    #     :param labels: labels
    #     :return: loss
    #     """

    #     # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
    #     target = - torch.ones_like(labels)
    #     for i, c in enumerate(self.valid_labels):
    #         target[labels == c] = i

    #     # Reshape to have a minibatch size of 1
    #     outputs = torch.transpose(outputs, 0, 1)
    #     outputs = outputs.unsqueeze(0)
    #     target = target.unsqueeze(0)

    #     # Cross entropy loss
    #     self.output_loss = self.criterion(outputs, target)

    #     # Regularization of deformable offsets
    #     if self.deform_fitting_mode == 'point2point':
    #         self.reg_loss = p2p_fitting_regularizer(self)
    #     elif self.deform_fitting_mode == 'point2plane':
    #         raise ValueError('point2plane fitting mode not implemented yet.')
    #     else:
    #         raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

    #     # Combined loss
    #     return self.output_loss + self.reg_loss

    # def accuracy(self, outputs, labels):
    #     """
    #     Computes accuracy of the current batch
    #     :param outputs: logits predicted by the network
    #     :param labels: labels
    #     :return: accuracy value
    #     """

    #     # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
    #     target = - torch.ones_like(labels)
    #     for i, c in enumerate(self.valid_labels):
    #         target[labels == c] = i

    #     predicted = torch.argmax(outputs.data, dim=1)
    #     total = target.size(0)
    #     correct = (predicted == target).sum().item()

    #     return correct / total

# class KPFCNN(nn.Module):
#     """
#     Class defining KPFCNN
#     """

#     def __init__(self, config, lbl_values, ign_lbls):
#         super(KPFCNN, self).__init__()

#         ############
#         # Parameters
#         ############

#         # Current radius of convolution and feature dimension
#         layer = 0
#         r = config.first_subsampling_dl * config.conv_radius
#         in_dim = config.in_features_dim # 2
#         out_dim = config.first_features_dim # 128
#         self.K = config.num_kernel_points
#         self.C = len(lbl_values) - len(ign_lbls)
#         self.task = config.dataset_task

#         #####################
#         # List Encoder blocks
#         #####################

#         # Save all block operations in a list of modules
#         self.encoder_blocks = nn.ModuleList()
#         self.temporal_convs = nn.ModuleList()
#         self.encoder_skip_dims = []
#         self.encoder_skips = []
#         self.curr_t      = config.architecture_t
#         # Loop over consecutive blocks
#         for block_i, block in enumerate(config.architecture):

#             # Check equivariance
#             if ('equivariant' in block) and (not out_dim % 3 == 0):
#                 raise ValueError('Equivariant block but features dimension is not a factor of 3')

#             # Detect change to next layer for skip connection
#             if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
#                 self.encoder_skips.append(block_i)
#                 self.encoder_skip_dims.append(in_dim)
#                 if self.curr_t[block_i] > 1: 
#                     self.temporal_convs.append(UnaryBlock(2*in_dim, in_dim, config.use_batch_norm, config.batch_norm_momentum, no_relu=False))

#             # Detect upsampling block to stop
#             if 'upsample' in block:
#                 break

#             # Apply the good block function defining tf ops
#             self.encoder_blocks.append(block_decider(block,
#                                                     r,
#                                                     in_dim,
#                                                     out_dim,
#                                                     layer,
#                                                     config,
#                                                     block_ind=block_i))

#             # Update dimension of input from output
#             if 'simple' in block:
#                 in_dim = out_dim // 2
#             else:
#                 in_dim = out_dim

#             # Detect change to a subsampled layer
#             if 'pool' in block or 'strided' in block:
#                 # Update radius and feature dimension for next layer
#                 layer += 1
#                 r *= 2
#                 out_dim *= 2

#         #####################
#         # List Decoder blocks
#         #####################

#         # Save all block operations in a list of modules
#         self.decoder_blocks = nn.ModuleList()
#         self.decoder_concats = []

#         # Find first upsampling block
#         start_i = 0
#         for block_i, block in enumerate(config.architecture):
#             if 'upsample' in block:
#                 start_i = block_i
#                 break

#         # Loop over consecutive blocks
#         for block_i, block in enumerate(config.architecture[start_i:]):

#             # Add dimension of skip connection concat
#             if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
#                 in_dim += self.encoder_skip_dims[layer]
#                 self.decoder_concats.append(block_i)

#             # Apply the good block function defining tf ops
#             self.decoder_blocks.append(block_decider(block,
#                                                     r,
#                                                     in_dim,
#                                                     out_dim,
#                                                     layer,
#                                                     config,
#                                                     block_ind=block_i))

#             # Update dimension of input from output
#             in_dim = out_dim

#             # Detect change to a subsampled layer
#             if 'upsample' in block:
#                 # Update radius and feature dimension for next layer
#                 layer -= 1
#                 r *= 0.5
#                 out_dim = out_dim // 2

#         self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
#         self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)

#         ################
#         # Network Losses
#         ################

#         # List of valid labels (those not ignored in loss)
#         self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

#         # Choose segmentation loss
#         #sscrit
#         if len(config.class_w) > 0:
#             class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
#             self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
#         else:
#             self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
#         self.deform_fitting_mode = config.deform_fitting_mode
#         self.deform_fitting_power = config.deform_fitting_power
#         self.deform_lr_factor = config.deform_lr_factor
#         self.repulse_extent = config.repulse_extent
#         self.output_loss = 0
#         self.reg_loss = 0
#         self.l1 = nn.L1Loss()

#         return

#     def forward(self, batch, config):

#         # Get input features
#         x = batch.features.clone().detach()

#         # Loop over consecutive blocks
#         skip_x = []
#         for block_i, block_op in enumerate(self.encoder_blocks):
#             if block_i in self.encoder_skips:
#                 skip_x.append(x)
#             x = block_op(x, batch)

#         for block_i, block_op in enumerate(self.decoder_blocks):
#             if block_i in self.decoder_concats:
#                 x = torch.cat([x, skip_x.pop()], dim=1)
#             x = block_op(x, batch)

#         # Head of network
#         x = self.head_mlp(x, batch)
#         x = self.head_softmax(x, batch)

#         return x
    # def forward(self, batch, config):

    #     # Get input features
    #     x = batch.features.clone().detach()
    #     # Loop over consecutive blocks
    #     # print(x.size())
    #     skip_x = []
    #     tid = 0
    #     for block_i, block_op in enumerate(self.encoder_blocks):
    #         if block_i in self.encoder_skips:
    #             x_seq = x.view(-1, self.curr_t[block_i] , x.size(-1)).contiguous()
    #             x_sum = x_seq[:, 0, :] + x_seq[:, 1, :]
    #             x_diff= x_seq[:, 0, :] - x_seq[:, 1, :]
    #             skip_x.append(self.temporal_convs[tid](torch.cat([x_sum, x_diff], dim=1)))
    #             tid+=1
    #             if self.curr_t[block_i+1] < 2: 
    #                 x = skip_x[-1]
    #         # print(block_i, block_op.layer_ind, block_op)
    #         print(f'block {block_i} has x is {x.size()}, neighbors {batch.neighbors[block_op.layer_ind].size()}, t_dim {self.curr_t[block_i]}')
    #         # breakpoint()
    #         x = block_op(x, batch)

    #     for block_i, block_op in enumerate(self.decoder_blocks):
    #         if block_i in self.decoder_concats:
    #             x = torch.cat([x, skip_x.pop()], dim=1)
    #         # breakpoint()
    #         x = block_op(x, batch)

    #     # Head of network
    #     x = self.head_mlp(x, batch)
    #     x = self.head_softmax(x, batch)

    #     return x

    #ssloss
    def loss(self, outputs, labels, m_pred, m_label):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)

        breakpoint()
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i  # loss here doesn't look right

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)
        self.motion_loss = self.criterion_m(m_pred, m_label)
        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        if self.task == 'motion_segmentation': 
            predicted = torch.argmax(outputs.data, dim=1).cpu().numpy()
            target = target.cpu().numpy()
            indb = np.where(target==-1)[0]
            ind0 = np.where(target==0)[0]
            ind1 = np.where(target==1)[0]
            # predicted, 
            total   = len(ind0) + len(ind1)
            correct0 = (predicted[ind0] == target[ind0]).sum()
            correct1 = (predicted[ind1] == target[ind1]).sum()
            correct  = correct0 + correct1
            print(f'motion class has ratio: {total/target.shape[0]}, correctness non-move: {correct0}/{len(ind0)}, moving: {correct1}/{len(ind1)}')
        else: 
            predicted = torch.argmax(outputs.data, dim=1)
            total = target.size(0)
            correct = (predicted == target).sum().item()
        return correct / (total+10e-9)





















