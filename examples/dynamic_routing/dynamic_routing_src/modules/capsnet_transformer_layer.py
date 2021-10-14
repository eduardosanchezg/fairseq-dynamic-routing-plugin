# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
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
        batch_size = x.size(0)

        x = x.transpose(1, 2) # dim 1 and dim 2 are swapped. out tensor shape: [128, 1152, 8]

        # Stacking and adding a dimension to a tensor.
        # stack ops output shape: [128, 1152, 10, 8]
        # unsqueeze ops output shape: [128, 1152, 10, 8, 1]
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)

        # Convert single weight to batch weight.
        # [1 x 1152 x 10 x 16 x 8] to: [128, 1152, 10, 16, 8]
        batch_weight = torch.cat([self.weight] * batch_size, dim=0)

        # u_hat is "prediction vectors" from the capsules in the layer below.
        # Transform inputs by weight matrix.
        # Matrix product of 2 tensors with shape: [128, 1152, 10, 16, 8] x [128, 1152, 10, 8, 1]
        # u_hat shape: [128, 1152, 10, 16, 1]
        batch_weight = batch_weight.cuda()
        x = x.cuda().float()

        print("|||||||||||||||||||||||||||||||||| D E B U G ||||||||||||||||||||||||||||||||||")
        print(batch_weight.size())
        print(x.size())
        u_hat = batch_weight @ x


        # All the routing logits (b_ij in the paper) are initialized to zero.
        # self.in_channel = primary_unit_size = 32 * 6 * 6 = 1152
        # self.num_unit = num_classes = 10
        # b_ij shape: [1, 1152, 10, 1]
        b_ij = Variable(torch.zeros(1, self.in_channel, self.num_unit, 1))
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
            # c_ij shape from: [128, 1152, 10, 1] to: [128, 1152, 10, 1, 1]
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Implement equation 2 in the paper.
            # s_j is total input to a capsule, is a weigthed sum over all "prediction vectors".
            # u_hat is weighted inputs, prediction ˆuj|i made by capsule i.
            # c_ij * u_hat shape: [128, 1152, 10, 16, 1]
            # s_j output shape: [batch_size=128, 1, 10, 16, 1]
            # Sum of Primary Capsules outputs, 1152D becomes 1D.
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # Squash the vector output of capsule j.
            # v_j shape: [batch_size, weighted sum of PrimaryCaps output,
            #             num_classes, output_unit_size from u_hat, 1]
            # == [128, 1, 10, 16, 1]
            # So, the length of the output vector of a capsule is 16, which is in dim 3.
            v_j = utils.squash(s_j, dim=3)

            # in_channel is 1152.
            # v_j1 shape: [128, 1152, 10, 16, 1]
            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)

            # The agreement.
            # Transpose u_hat with shape [128, 1152, 10, 16, 1] to [128, 1152, 10, 1, 16],
            # so we can do matrix product u_hat and v_j1.
            # u_vj1 shape: [1, 1152, 10, 1]
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update routing (b_ij) by adding the agreement to the initial logit.
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1) # shape: [128, 10, 16, 1]

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
        return utils.squash(unit, dim=2) # dim 2 is the third dim (1152D array) in our tensor

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

        IN_UNIT = 1
        IN_CHANNEL = 512
        NUM_UNIT = 3
        UNIT_SIZE = 4
        NUM_ROUTING = 5

        capsnet = CapsuleSubLayer(in_unit=IN_UNIT, in_channel=IN_CHANNEL, num_unit=NUM_UNIT, unit_size=UNIT_SIZE, use_routing=True, num_routing=NUM_ROUTING, cuda_enabled=True)

        x = capsnet.forward(x)

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
