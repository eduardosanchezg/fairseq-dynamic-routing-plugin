# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
from fairseq.modules import TransformerEncoderLayer
from torch import Tensor
import torch
from torch import nn

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
        self.head_dim = 32 # todo: generalize
        self.num_heads = 16 # todo: generalize
        self.in_dim = 32
        self.out_dim = 32
        self.num_in = 16
        self.num_out = 16
        self.dynamic_routing_weights = nn.Parameter(torch.empty( (self.num_in, self.num_out, self.in_dim, self.out_dim), device='cuda', requires_grad=True)).requires_grad_()
        print("INITIALIZED!!!!!!")


        self.self_attn_layer = ModifiedMultiheadAttention(
            self.embed_dim,
            #self.cfg.encoder.attention_heads,
            #dropout=self.cfg.attention_dropout,
            num_heads=16,
            dropout=0.3,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            dynamic_routing_weights = self.dynamic_routing_weights
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

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        self.embed_dim = x.size(2)

        x, _ = self.self_attn_layer.forward(
             query=x,
             key=x,
             value=x,
             key_padding_mask=encoder_padding_mask,
             need_weights=False,
             attn_mask=attn_mask,
         )

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

        # print("|||||||||||||||||||||||||||||||||| FINAL || OUTPUT ||||||||||||||||||||||||||||||||||")
        # print(x.size())
        # print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||")

        print("w: {}".format(str(self.dynamic_routing_weights.abs().mean())))
        if self.dynamic_routing_weights.grad != None:
            print("grad: {}".format(self.dynamic_routing_weights.grad.norm()))
        else:
            print("grad: None")
        print(self.dynamic_routing_weights[0,0,0,:])

        return x


