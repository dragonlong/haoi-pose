#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling datasets
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import os
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader, Dataset

import __init__
from common.mayavi_visu import *

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors

def breakpoint():
    import pdb;pdb.set_trace()
# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#
def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/


class PointCloudDataset(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, name, config):
        """
        Initialize parameters of the dataset here.
        """

        self.name = name
        self.path = ''
        self.label_to_names = {}
        self.motion_labels_range = {}
        self.num_classes = 0
        self.num_motion_classes = 0
        self.label_values = np.zeros((0,), dtype=np.int32)
        self.label_names = []
        self.label_to_idx = {}
        self.name_to_label = {}
        self.config = config
        self.neighborhood_limits = []

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return 0

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """

        return 0

    def init_labels(self):

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values= np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx  = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.num_motion_classes  = len(self.label_to_motion)
        self.motion_label_values = np.sort([k for k, v in self.label_to_motion.items()])

    def augmentation_transform(self, points, normals=None, verbose=False, return_components_only=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])
        if self.set in ['validation', 'test']:
            if normals is None:
                return points, 1, R
                
        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Choose two random angles for the first vector in polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle
                alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            if return_components_only: 
                return noise, scale, R
            else: 
                return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def classification_inputs(self,
                              stacked_points,
                              stacked_features,
                              labels,
                              stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 1), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # Save deform layers

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_stack_lengths
        li += [stacked_features, labels]

        return li


    def segmentation_inputs(self,
                            stacked_points,
                            stacked_features,
                            labels,
                            stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points)+1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths
        li += [stacked_features, labels]

        return li

    def segmentation_inputs_seq(self,
                            stacked_points,
                            stacked_features,
                            labels,
                            stack_lengths):

        n_frame= self.config.n_frames
        w_size = self.config.window_size
        arch = self.config.architecture

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_querys = []
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################
        query_points  = None
        support_lengths = stack_lengths
        support_points  = stacked_points
        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************
            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # get query points if None
                if query_points is None:
                    query_points = np.zeros(( w_size * sum(support_lengths), 3), dtype=np.float32)
                    query_lengths= []
                    for j in range(0, len(support_lengths), w_size):
                        s_ind, e_ind  = sum( support_lengths[0:j] ), sum( support_lengths[0:j+w_size] )
                        frame_points  = np.tile( support_points[s_ind:e_ind], [w_size, 1])
                        query_points[w_size*s_ind:w_size*e_ind] = frame_points
                        query_lengths += [e_ind - s_ind] * w_size
                # print(f'support lengths: {support_lengths}, query_lengths: {query_lengths}')
                conv_i = batch_neighbors(query_points, support_points, query_lengths, support_lengths, r)
            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************
            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(support_points, support_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                pool_points = np.zeros(( w_size * sum(pool_b), 3), dtype=np.float32)
                pool_lengths= []
                for j in range(0, len(pool_b), w_size):
                    s_ind, e_ind  = sum( pool_b[0:j] ), sum( pool_b[0:j+w_size] )
                    frame_points  = np.tile( pool_p[s_ind:e_ind], [w_size, 1])
                    pool_points[w_size*s_ind:w_size*e_ind] = frame_points
                    pool_lengths += [e_ind - s_ind] * w_size

                pool_i = batch_neighbors(pool_points, support_points, pool_lengths, support_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(support_points, pool_p, support_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points)+1)

            # Updating input lists
            input_querys += [query_points]
            input_points += [support_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [support_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            support_points  = pool_p
            support_lengths = pool_b
            query_points=pool_points
            query_lengths=pool_lengths

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        # list of network inputs
        li = input_querys + input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths
        li += [stacked_features, labels]

        save_viz = False
        if save_viz:
            save_path = join(self.config.log_dir, 'viz')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_name = save_path + f'/{self.epoch_i.cpu().numpy()[0]}.npy'
            viz_dict = {}
            viz_dict['input_querys'] = input_querys
            viz_dict['input_points'] = input_points 
            viz_dict['input_neighbors'] = input_neighbors
            viz_dict['input_pools']  = input_pools
            viz_dict['input_upsamples'] = input_upsamples
            viz_dict['input_stack_lengths'] = input_stack_lengths
            viz_dict['label'] = labels

            for key, value in viz_dict.items():
                print(key, len(value))

            print('saving to ', save_name)
            np.save(save_name, arr=viz_dict)
        return li


    # def segmentation_inputs_video(self,
    #                         query_points,
    #                         stacked_points,
    #                         stacked_features,
    #                         labels,
    #                         query_lengths,
    #                         stack_lengths):

    #     # Starting radius of convolutions
    #     r_normal = self.config.first_subsampling_dl * self.config.conv_radius

    #     # Starting layer
    #     layer_blocks = []

    #     # Lists of inputs
    #     input_points_full  = [query_points, stacked_points] # contain initial full query points + support points
    #     input_neighbors_full = []
    #     input_points = []
    #     input_neighbors = []
    #     input_pools = []
    #     input_upsamples = []
    #     input_stack_lengths = []
    #     deform_layers = []
    #     batch_size = len(query_lengths)
    #     query_points = query_points.astype(np.float32)
    #     stacked_points = stacked_points.astype(np.float32)

    #     ######################
    #     # Loop over the blocks
    #     ######################

    #     arch = self.config.architecture
    #     # input_points_full += [stacked_points]
    #     # input_stack_lengths += [stack_lengths]
    #     for block_i, block in enumerate(arch):
    #         # breakpoint()
    #         # Get all blocks of the layer
    #         if block_i == 0: 
    #             r = r_normal # use rigid kernel by default
    #             conv_i = batch_neighbors(query_points, stacked_points, query_lengths, stack_lengths, r)#TODO
    #             conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
    #             input_neighbors_full += [conv_i.astype(np.int64)]

    #         if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
    #             layer_blocks += [block]
    #             continue

    #         # Convolution neighbors indices
    #         # *****************************
    #         prev_t_dim = self.config.architecture_t[block_i-1]
    #         curr_t_dim = self.config.architecture_t[block_i]
    #         deform_layer = False
    #         if layer_blocks:
    #             # Convolutions are done in this layer, compute the neighbors with the good radius
    #             if np.any(['deformable' in blck for blck in layer_blocks]):
    #                 r = r_normal * self.config.deform_radius / self.config.conv_radius
    #                 deform_layer = True
    #             else:
    #                 r = r_normal
    #             query_points  = query_points[:query_lengths[0] * prev_t_dim]
    #             query_lengths = query_lengths[:prev_t_dim]
    #             conv_i = batch_neighbors(query_points, query_points, query_lengths, query_lengths, r)
    #         else:
    #             # This layer only perform pooling, no neighbors required
    #             conv_i = np.zeros((0, 1), dtype=np.int32)

    #         # Pooling neighbors indices
    #         # *************************

    #         # If end of layer is a pooling operation
    #         if 'pool' in block or 'strided' in block:
    #             query_points  = query_points[:query_lengths[0] * curr_t_dim] # pool always cares current time_dim, 
    #             query_lengths = query_lengths[:curr_t_dim]
    #             # New subsampling length
    #             dl = 2 * r_normal / self.config.conv_radius

    #             # Subsampled points
    #             pool_p, pool_b = batch_grid_subsampling(query_points, query_lengths, sampleDl=dl) #TODO

    #             # Radius of pooled neighbors
    #             if 'deformable' in block:
    #                 r = r_normal * self.config.deform_radius / self.config.conv_radius
    #                 deform_layer = True
    #             else:
    #                 r = r_normal

    #             # Subsample indices
    #             pool_i = batch_neighbors(pool_p, query_points, pool_b, query_lengths, r)

    #             # Upsample indices (with the radius of the next layer to keep wanted density)
    #             # breakpoint()
    #             up_i = batch_neighbors(query_points[:query_lengths[0]], pool_p[:pool_b[0]], query_lengths[0:1], pool_b[0:1], 2 * r)

    #         else:
    #             # No pooling in the end of this layer, no pooling indices required
    #             pool_i = np.zeros((0, 1), dtype=np.int32)
    #             pool_p = np.zeros((0, 3), dtype=np.float32)
    #             pool_b = np.zeros((0,), dtype=np.int32)
    #             up_i   = np.zeros((0, 1), dtype=np.int32)

    #         # Reduce size of neighbors matrices by eliminating furthest point
    #         conv_i = self.big_neighborhood_filter(conv_i, len(input_points)) # TODO, fold
    #         pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
    #         if up_i.shape[0] > 0:
    #             up_i = self.big_neighborhood_filter(up_i, len(input_points)+1)

    #         # Updating input lists
    #         input_points += [query_points]
    #         input_stack_lengths += [query_lengths]            
    #         # breakpoint()   
    #         # print(f'block {block_i}, {block} has neighbors {conv_i.shape}, pooled neighbors{pool_i.shape}')
    #         input_neighbors += [conv_i.astype(np.int64)]
    #         input_pools += [pool_i.astype(np.int64)]
    #         input_upsamples += [up_i.astype(np.int64)]
    #         deform_layers += [deform_layer]

    #         # New points for next layer
    #         query_points = pool_p
    #         query_lengths = pool_b

    #         # Update radius and reset blocks
    #         r_normal *= 2
    #         layer_blocks = []

    #         # Stop when meeting a global pooling or upsampling
    #         if 'global' in block or 'upsample' in block:
    #             break

    #     ###############
    #     # Return inputs
    #     ###############

    #     # list of network inputs
    #     li = input_points_full + input_neighbors_full + input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths
    #     # for i, item in enumerate(input_points):
    #     #     647(f'{i} points has {item.shape}')
    #     # for i, item in enumerate(input_neighbors):
    #     #     print(f'{i} neighbors has {item.shape}')
    #     # for i, item in enumerate(input_pools):
    #     #     print(f'{i} neighbors for pooled has {item.shape}')
    #     li += [stacked_features, labels]

    #     return li










