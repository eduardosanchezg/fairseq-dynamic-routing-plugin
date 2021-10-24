# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

        print("///////////////////////////////////////////////////////// LAYERS ///////////////////////////////////////////////")
        print(self.layers)
        print("///////////////////////////////////////////////////////////////////////////////////////")
        from torchsummary import summary
        summary(self, (128, 224, 512))





@register_model_architecture(
    "capsnet_transformer", "capsnet_transformer"
)
def capsnet_transformer_architecture(args):
    base_architecture(args)