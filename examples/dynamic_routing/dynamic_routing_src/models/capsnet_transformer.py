# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
)

from fairseq.models.transformer import TransformerModel
from ..modules.capsnet_transformer_layer import CapsNetTransformerEncoderLayer


@register_model("capsnet_transformer")
class CapsNetTransformerModel(TransformerModel):
    """TODO: A variant of Transformer as is in "xxx"
    (https://arxiv.org/abs/2009.13102).
    """

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return CapsNetTransformerEncoder(args, src_dict, embed_tokens)



class CapsNetTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.layers[0] = CapsNetTransformerEncoderLayer(args)
        # clip_value = 1
        # for p in self.layers[0].parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))




@register_model_architecture(
    "capsnet_transformer", "capsnet_transformer"
)
def capsnet_transformer_architecture(args):
    base_architecture(args)