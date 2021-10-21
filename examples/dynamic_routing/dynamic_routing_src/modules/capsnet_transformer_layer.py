# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import Parameter

from .dynamic_routing import DynamicRouting

#from examples.dynamic_routing.dynamic_routing_src.modules.dynamic_routing import DynamicRouting
from fairseq.modules import TransformerEncoderLayer
from torch import Tensor
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from .modified_multihead_attention import ModifiedMultiheadAttention


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
    capsule_proj_weight: Tensor
    capsule_proj_bias: Tensor
    def __init__(self, args):
        super().__init__(args)
        self.capsule_proj_weight = Parameter(torch.randn(512, 16))
        self.capsule_proj_bias = Parameter(torch.randn(512))

        # print("||||||||||||||TENSOR CHECK (capsnet_trans init)||||||||||")
        # print(self.capsule_proj_weight.size())
        # print(self.capsule_proj_bias.size())

        self.self_attn_layer = ModifiedMultiheadAttention(
            self.embed_dim,
            #self.cfg.encoder.attention_heads,
            #dropout=self.cfg.attention_dropout,
            num_heads=16,
            dropout=0.1,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            capsule_proj_weight=self.capsule_proj_weight,
            capsule_proj_bias=self.capsule_proj_bias
        )



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
        # print("||||||||||||||||RESIDUAL|||||||||||||||||||||")
        # print(x.size())
        # print("||||||||||||||||||||||")
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # print("||||||||||||||||NORMALIZED|||||||||||||||||||||")
        # print(x.size())
        # print("||||||||||||||||||||||")

        self.embed_dim = x.size(2)


        #self_attn_layer = self_attn_layer.half()
        #self_attn_layer = self.self_attn_layer.cuda()
        x, _ = self.self_attn_layer(
             query=x,
             key=x,
             value=x,
             key_padding_mask=encoder_padding_mask,
             need_weights=False,
             attn_mask=attn_mask,
         )

        # x, _ = self.self_attn(
        #     query=x,
        #     key=x,
        #     value=x,
        #     key_padding_mask=encoder_padding_mask,
        #     need_weights=False,
        #     attn_mask=attn_mask,
        # )

        # print("||||||||||||||||AFTER SELF-ATTENTION|||||||||||||||||||||")
        # print(x.size())
        # print("||||||||||||||||||||||")
        # print("||||||||||||||attn weights||||||||")
        # print(attn.size())
        # print("|||||||||||||||||||")
        IN_UNIT = 224
        IN_CHANNEL = 2
        NUM_UNIT = 224
        UNIT_SIZE = 512
        NUM_ROUTING = 3

        #capsnet = CapsuleSubLayer(in_unit=IN_UNIT, in_channel=IN_CHANNEL, num_unit=NUM_UNIT, unit_size=UNIT_SIZE, use_routing=True, num_routing=NUM_ROUTING, cuda_enabled=True)

        #x = capsnet.forward(x)

        #print("|||||||||||||||||||||||||||||||||| D E B U G || OUTER LAYER ||||||||||||||||||||||||||||||||||")
        #print(residual.size())
        #print(x.size())

       # dynamic_routing = DynamicRouting(12,512,False)

        #x = dynamic_routing.forward(x,3)

        x = self.dropout_module(x)

        # print("|||||||||||||||||||||||||||||||||| D E B U G || OUTER LAYER ||||||||||||||||||||||||||||||||||")
        # print(residual.size())
        # print(x.size())
        # print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||")


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

        # print("|||||||||||||||||||||||||||||||||| FINAL || OUTPUT ||||||||||||||||||||||||||||||||||")
        # print(x.size())
        # print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||")

        return x


