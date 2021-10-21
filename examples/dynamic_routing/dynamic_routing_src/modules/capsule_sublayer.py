import torch
from torch import nn

from .utils import squash

BATCH_SIZE = 100
NUM_CLASSES = 10
#NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3

from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm
import torchnet as tnt
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


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
        self.unit_size = unit_size
        if self.use_routing:
            """
            Based on the paper, DigitCaps which is capsule layer(s) with
            capsule inputs use a routing algorithm that uses this weight matrix, Wij
            """
            # weight shape:
            # [1 x primary_unit_size x num_classes x output_unit_size x num_primary_unit]
            # == [1 x 1152 x 10 x 16 x 8]
            self.weight = nn.Parameter(torch.randn(1 ,unit_size, num_unit,in_channel , in_unit))
            self.weight.requires_grad = False
        else:
            """
            
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

        # print("||||INITIAL VALUE|||")
        # print(x.size())
        # print("||||||||||||||||||||")

        batch_size = x.size(0)

        x = x.transpose(1, 2) # dim 1 and dim 2 are swapped. out tensor shape: [128, 1152, 8]

        # print("||||AFTER TRANSPOSE|||")
        # print(x.size())
        # print("||||||||||||||||||||")

        # Stacking and adding a dimension to a tensor.
        # stack ops output shape: [128, 1152, 10, 8]
        # unsqueeze ops output shape: [128, 1152, 10, 8, 1]
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)

        # print("||||AFTER STACKING AND ADDING|||")
        # print(x.size())
        # print("||||||||||||||||||||")

        # Convert single weight to batch weight.
        # [1 x 1152 x 10 x 16 x 8] to: [128, 1152, 10, 16, 8]
        batch_weight = torch.cat([self.weight] * batch_size, dim=0)

        # u_hat is "prediction vectors" from the capsules in the layer below.
        # Transform inputs by weight matrix.
        # Matrix product of 2 tensors with shape: [128, 1152, 10, 16, 8] x [128, 1152, 10, 8, 1]
        # u_hat shape: [128, 1152, 10, 16, 1]
        batch_weight = batch_weight

        # print("|||||||||||||||||||||||||||||||||| D E B U G || INNER LAYER ||||||||||||||||||||||||||||||||||")
        # print(batch_weight.size())
        # print(x.size())

        u_hat = torch.matmul(batch_weight.cuda().half(), x.half())

        # print("|||||||||||||||||||||||||||||||||| U HAT ||||||||||||||||||||||||||||||||||")
        # print(u_hat.size())

        # All the routing logits (b_ij in the paper) are initialized to zero.
        # self.in_channel = primary_unit_size = 32 * 6 * 6 = 1152
        # self.num_unit = num_classes = 10
        # b_ij shape: [1, 1152, 10, 1]
        b_ij = Variable(torch.zeros(1, self.unit_size, self.num_unit, 1))

        # print("|||||||||||||||||||||||||||||||||| Bij ||||||||||||||||||||||||||||||||||")
        # print(b_ij.size())

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

            # print("||||Cij AFTER SOFTMAX|||")
            # print(c_ij.size())
            # print(iteration)
            # print("||||||||||||||||||||")

            # c_ij shape from: [128, 1152, 10, 1] to: [128, 1152, 10, 1, 1]
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # print("||||Cij AFTER UNZQUEEZE|||")
            # print(c_ij.size())
            # print(iteration)
            # print("||||||||||||||||||||")

            # Implement equation 2 in the paper.
            # s_j is total input to a capsule, is a weigthed sum over all "prediction vectors".
            # u_hat is weighted inputs, prediction ˆuj|i made by capsule i.
            # c_ij * u_hat shape: [128, 1152, 10, 16, 1]
            # s_j output shape: [batch_size=128, 1, 10, 16, 1]
            # Sum of Primary Capsules outputs, 1152D becomes 1D.
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # print("||||Sj AFTER MUL AND SUM|||")
            # print(s_j.size())
            # print(iteration)
            # print("||||||||||||||||||||")

            # Squash the vector output of capsule j.
            # v_j shape: [batch_size, weighted sum of PrimaryCaps output,
            #             num_classes, output_unit_size from u_hat, 1]
            # == [128, 1, 10, 16, 1]
            # So, the length of the output vector of a capsule is 16, which is in dim 3.
            v_j = squash(s_j, dim=3)

            # print("||||Vj after squash|||")
            # print(v_j.size())
            # print(iteration)
            # print("||||||||||||||||||||")

            # in_channel is 1152.
            # v_j1 shape: [128, 1152, 10, 16, 1]
            v_j1 = torch.cat([v_j] * self.unit_size, dim=1)

            # print("||||v_j1 after cat|||")
            # print(v_j1.size())
            # print(iteration)
            # print("||||||||||||||||||||")

            # The agreement.
            # Transpose u_hat with shape [128, 1152, 10, 16, 1] to [128, 1152, 10, 1, 16],
            # so we can do matrix product u_hat and v_j1.
            # u_vj1 shape: [1, 1152, 10, 1]
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1.half()).squeeze(4).mean(dim=0, keepdim=True)

            # print("||||u_vj1 after matmul|||")
            # print(u_vj1.size())
            # print(iteration)
            # print("||||||||||||||||||||")

            # Update routing (b_ij) by adding the agreement to the initial logit.

            # print("||||Bij and Uvj1 before adding|||")
            # print(b_ij.size())
            # print(u_vj1.size())
            # print(iteration)
            # print("||||||||||||||||||||")

            b_ij = b_ij + u_vj1

        #ORIGINAL: return v_j.squeeze(1) # shape: [128, 10, 16, 1]
        squeezed =  v_j.squeeze(1).squeeze(3) # shape: [128, 10, 16, 1]

        print("||||squeezed and normal|||")
        print(squeezed.size())
        print(v_j.size())
        print("||||||||||||||||||||")

        return squeezed

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