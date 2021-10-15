# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq.modules import TransformerEncoderLayer
from torch import Tensor
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3

from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm
import torchnet as tnt


class CapsuleSubLayer(nn.Module):
    """
    The core implementation of the idea of capsules
    """

    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing,
                 num_routing, cuda_enabled):
        super(CapsuleSubLayer, self).__init__()

        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.use_routing = use_routing
        self.num_routing = num_routing
        self.cuda_enabled = cuda_enabled

        if self.use_routing:
            """
            Based on the paper, DigitCaps which is capsule layer(s) with
            capsule inputs use a routing algorithm that uses this weight matrix, Wij
            """
            # weight shape:
            # [1 x primary_unit_size x num_classes x output_unit_size x num_primary_unit]
            # == [1 x 1152 x 10 x 16 x 8]
            self.weight = nn.Parameter(torch.randn(1, in_channel, num_unit, unit_size, in_unit))
        else:
            """
            According to the CapsNet architecture section in the paper,
            we have routing only between two consecutive capsule layers (e.g. PrimaryCapsules and DigitCaps).
            No routing is used between Conv1 and PrimaryCapsules.
            This means PrimaryCapsules is composed of several convolutional units.
            """
            # Define 8 convolutional units.
            self.conv_units = nn.ModuleList([
                nn.Conv2d(self.in_channel, 32, 9, 2) for u in range(self.num_unit)
            ])

    def forward(self, x):
        if self.use_routing:
            # Currently used by DigitCaps layer.
            return self.routing(x)
        else:
            # Currently used by PrimaryCaps layer.
            return self.no_routing(x)

    def routing(self, x):
        """
        Routing algorithm for capsule.
        :input: tensor x of shape [128, 8, 1152]
        :return: vector output of capsule j
        """

        print("||||INITIAL VALUE|||")
        print(x.size())
        print("||||||||||||||||||||")

        batch_size = x.size(0)

        x = x.transpose(1, 2) # dim 1 and dim 2 are swapped. out tensor shape: [128, 1152, 8]

        print("||||AFTER TRANSPOSE|||")
        print(x.size())
        print("||||||||||||||||||||")

        # Stacking and adding a dimension to a tensor.
        # stack ops output shape: [128, 1152, 10, 8]
        # unsqueeze ops output shape: [128, 1152, 10, 8, 1]
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)

        print("||||AFTER STACKING AND ADDING|||")
        print(x.size())
        print("||||||||||||||||||||")

        # Convert single weight to batch weight.
        # [1 x 1152 x 10 x 16 x 8] to: [128, 1152, 10, 16, 8]
        batch_weight = torch.cat([self.weight] * batch_size, dim=0)

        # u_hat is "prediction vectors" from the capsules in the layer below.
        # Transform inputs by weight matrix.
        # Matrix product of 2 tensors with shape: [128, 1152, 10, 16, 8] x [128, 1152, 10, 8, 1]
        # u_hat shape: [128, 1152, 10, 16, 1]
        batch_weight = batch_weight.cuda()
        x = x.cuda().float()

        print("|||||||||||||||||||||||||||||||||| D E B U G || INNER LAYER ||||||||||||||||||||||||||||||||||")
        print(batch_weight.size())
        print(x.size())

        u_hat = torch.matmul(batch_weight, x)

        print("|||||||||||||||||||||||||||||||||| U HAT ||||||||||||||||||||||||||||||||||")
        print(u_hat.size())
        # All the routing logits (b_ij in the paper) are initialized to zero.
        # self.in_channel = primary_unit_size = 32 * 6 * 6 = 1152
        # self.num_unit = num_classes = 10
        # b_ij shape: [1, 1152, 10, 1]
        b_ij = Variable(torch.zeros(1, self.in_channel, self.num_unit, 1))

        print("|||||||||||||||||||||||||||||||||| U HAT ||||||||||||||||||||||||||||||||||")
        print(b_ij.size())

        if self.cuda_enabled:
            b_ij = b_ij.cuda()

        # From the paper in the "Capsules on MNIST" section,
        # the sample MNIST test reconstructions of a CapsNet with 3 routing iterations.
        num_iterations = self.num_routing

        for iteration in range(num_iterations):
            # Routing algorithm

            # Calculate routing or also known as coupling coefficients (c_ij).
            # c_ij shape: [1, 1152, 10, 1]
            c_ij = F.softmax(b_ij, dim=2)  # Convert routing logits (b_ij) to softmax.

            print("||||Cij AFTER SOFTMAX|||")
            print(c_ij.size())
            print(iteration)
            print("||||||||||||||||||||")

            # c_ij shape from: [128, 1152, 10, 1] to: [128, 1152, 10, 1, 1]
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            print("||||Cij AFTER UNZQUEEZE|||")
            print(c_ij.size())
            print(iteration)
            print("||||||||||||||||||||")

            # Implement equation 2 in the paper.
            # s_j is total input to a capsule, is a weigthed sum over all "prediction vectors".
            # u_hat is weighted inputs, prediction ˆuj|i made by capsule i.
            # c_ij * u_hat shape: [128, 1152, 10, 16, 1]
            # s_j output shape: [batch_size=128, 1, 10, 16, 1]
            # Sum of Primary Capsules outputs, 1152D becomes 1D.
            s_j = (c_ij * u_hat).sum(dim=0, keepdim=True)

            print("||||Sj AFTER MUL AND SUM|||")
            print(s_j.size())
            print(iteration)
            print("||||||||||||||||||||")

            # Squash the vector output of capsule j.
            # v_j shape: [batch_size, weighted sum of PrimaryCaps output,
            #             num_classes, output_unit_size from u_hat, 1]
            # == [128, 1, 10, 16, 1]
            # So, the length of the output vector of a capsule is 16, which is in dim 3.
            v_j = squash(s_j, dim=3)

            print("||||Vj after squash|||")
            print(v_j.size())
            print(iteration)
            print("||||||||||||||||||||")

            # in_channel is 1152.
            # v_j1 shape: [128, 1152, 10, 16, 1]
            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)

            print("||||v_j1 after cat|||")
            print(v_j1.size())
            print(iteration)
            print("||||||||||||||||||||")

            # The agreement.
            # Transpose u_hat with shape [128, 1152, 10, 16, 1] to [128, 1152, 10, 1, 16],
            # so we can do matrix product u_hat and v_j1.
            # u_vj1 shape: [1, 1152, 10, 1]
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            print("||||u_vj1 after matmul|||")
            print(u_vj1.size())
            print(iteration)
            print("||||||||||||||||||||")

            # Update routing (b_ij) by adding the agreement to the initial logit.

            print("||||Bij and Uvj1 after matmul|||")
            print(b_ij.size())
            print(u_vj1.size())
            print(iteration)
            print("||||||||||||||||||||")

            b_ij = b_ij + u_vj1

        return v_j.squeeze() # shape: [128, 10, 16, 1]

    def no_routing(self, x):
        """
        Get output for each unit.
        A unit has batch, channels, height, width.
        An example of a unit output shape is [128, 32, 6, 6]
        :return: vector output of capsule j
        """
        # Create 8 convolutional unit.
        # A convolutional unit uses normal convolutional layer with a non-linearity (squash).
        unit = [self.conv_units[i](x) for i, l in enumerate(self.conv_units)]

        # Stack all unit outputs.
        # Stacked of 8 unit output shape: [128, 8, 32, 6, 6]
        unit = torch.stack(unit, dim=1)

        batch_size = x.size(0)

        # Flatten the 32 of 6x6 grid into 1152.
        # Shape: [128, 8, 1152]
        unit = unit.view(batch_size, self.num_unit, -1)

        # Add non-linearity
        # Return squashed outputs of shape: [128, 8, 1152]
        return squash(unit, dim=2) # dim 2 is the third dim (1152D array) in our tensor

class CapsNetTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block. TODO: comment

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__(args)
        #print('****************************', drop_residual_after_att)
        #self.drop_residual_after_att = drop_residual_after_att

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        #if attn_mask is not None:
        #    attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # x, _ = self.self_attn(
        #     query=x,
        #     key=x,
        #     value=x,
        #     key_padding_mask=encoder_padding_mask,
        #     need_weights=False,
        #     attn_mask=attn_mask,
        # )

        IN_UNIT = 224
        IN_CHANNEL = 512
        NUM_UNIT = 8
        UNIT_SIZE = 4
        NUM_ROUTING = 3

        capsnet = CapsuleSubLayer(in_unit=IN_UNIT, in_channel=IN_CHANNEL, num_unit=NUM_UNIT, unit_size=UNIT_SIZE, use_routing=True, num_routing=NUM_ROUTING, cuda_enabled=True)

        x = capsnet.forward(x)

        print("|||||||||||||||||||||||||||||||||| D E B U G || OUTER LAYER ||||||||||||||||||||||||||||||||||")
        print(residual.size())
        print(x.size())

        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x



"""Utilities
PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829
Author: Cedric Chee
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse


def one_hot_encode(target, length):
    """Converts batches of class indices to classes of one-hot vectors."""
    batch_s = target.size(0)
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec


def checkpoint(state, epoch):
    """Save checkpoint"""
    model_out_path = 'results/trained_model/model_epoch_{}.pth'.format(epoch)
    torch.save(state, model_out_path)
    print('Checkpoint saved to {}'.format(model_out_path))





def squash(sj, dim=2):
    """
    The non-linear activation used in Capsule.
    It drives the length of a large vector to near 1 and small vector to 0
    This implement equation 1 from the paper.
    """
    sj_mag_sq = torch.sum(sj**2, dim, keepdim=True)
    # ||sj||
    sj_mag = torch.sqrt(sj_mag_sq)
    v_j = (sj_mag_sq / (1.0 + sj_mag_sq)) * (sj / sj_mag)
    return v_j


def mask(out_digit_caps, cuda_enabled=True):
    """
    In the paper, they mask out all but the activity vector of the correct digit capsule.
    This means:
    a) during training, mask all but the capsule (1x16 vector) which match the ground-truth.
    b) during testing, mask all but the longest capsule (1x16 vector).
    Args:
        out_digit_caps: [batch_size, 10, 16] Tensor output of `DigitCaps` layer.
    Returns:
        masked: [batch_size, 10, 16, 1] The masked capsules tensors.
    """
    # a) Get capsule outputs lengths, ||v_c||
    v_length = torch.sqrt((out_digit_caps**2).sum(dim=2))

    # b) Pick out the index of longest capsule output, v_length by
    # masking the tensor by the max value in dim=1.
    _, max_index = v_length.max(dim=1)
    max_index = max_index.data

    # Method 1: masking with y.
    # c) In all batches, get the most active capsule
    # It's not easy to understand the indexing process with max_index
    # as we are 3D animal.
    batch_size = out_digit_caps.size(0)
    masked_v = [None] * batch_size # Python list
    for batch_ix in range(batch_size):
        # Batch sample
        sample = out_digit_caps[batch_ix]

        # Masks out the other capsules in this sample.
        v = Variable(torch.zeros(sample.size()))
        if cuda_enabled:
            v = v.cuda()

        # Get the maximum capsule index from this batch sample.
        max_caps_index = max_index[batch_ix]
        v[max_caps_index] = sample[max_caps_index]
        masked_v[batch_ix] = v # append v to masked_v

    # Concatenates sequence of masked capsules tensors along the batch dimension.
    masked = torch.stack(masked_v, dim=0)

    return masked


def accuracy(output, target, cuda_enabled=True):
    """
    Compute accuracy.
    Args:
        output: [batch_size, 10, 16, 1] The output from DigitCaps layer.
        target: [batch_size] Labels for dataset.
    Returns:
        accuracy (float): The accuracy for a batch.
    """
    batch_size = target.size(0)

    v_length = torch.sqrt((output**2).sum(dim=2, keepdim=True))
    softmax_v = F.softmax(v_length, dim=1)
    assert softmax_v.size() == torch.Size([batch_size, 10, 1, 1])

    _, max_index = softmax_v.max(dim=1)
    assert max_index.size() == torch.Size([batch_size, 1, 1])

    pred = max_index.squeeze() #max_index.view(batch_size)
    assert pred.size() == torch.Size([batch_size])

    if cuda_enabled:
        target = target.cuda()
        pred = pred.cuda()

    correct_pred = torch.eq(target, pred.data) # tensor
    # correct_pred_sum = correct_pred.sum() # scalar. e.g: 6 correct out of 128 images.
    acc = correct_pred.float().mean() # e.g: 6 / 128 = 0.046875

    return acc


def to_np(param):
    """
    Convert values of the model parameters to numpy.array.
    """
    return param.clone().cpu().data.numpy()


def str2bool(v):
    """
    Parsing boolean values with argparse.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')