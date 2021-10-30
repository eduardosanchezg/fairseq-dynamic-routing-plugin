import torch
from torch import nn

from .utils import squash



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
    num_in : int
    num_out : int
    in_dim : int
    out_dim : int
    def __init__(self, num_routing, cuda_enabled, weights, num_in, num_out, in_dim, out_dim):
        super(CapsuleSubLayer, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_routing = num_routing
        self.cuda_enabled = cuda_enabled
        self.weights = weights

    def forward(self, x):
        return self.routing(x)

    def routing(self, x):
        num_in, bsz, seq_len, in_dim = x.size()
        assert in_dim == self.in_dim
        assert num_in == self.num_in
        out_dim = self.out_dim
        num_out = self.num_out
        joint_batch = bsz * seq_len

        u = x.transpose(0,2).contiguous().view(joint_batch,num_in,in_dim) # [joint_batch, num_in, in_dim]

        stacked_u = torch.stack([u] * num_out, dim=2) # [joint_batch, num_in, num_out, in_dim]

        u_hat = torch.einsum('wxyz,bwxy->bwxz', (self.weights, stacked_u))

        #u_hat = torch.bmm(self.weights.expand(joint_batch, num_in, num_out, in_dim, out_dim).contiguous().view(joint_batch,num_in*num_out), stacked_u.unsqueeze(3)) # [joint_batch, num_in, num_out, out_dim]

        s = [None for _ in range(num_out)] # [out_dim]
        v = [None for _ in range(num_out)] # [out_dim]
        B = Variable(torch.zeros(num_in, num_out) )



        if self.cuda_enabled:
            B = B.cuda()
        num_iterations = self.num_routing

        for iteration in range(num_iterations):
            # Routing algorithm

            C = F.softmax(B, dim=1)  # Convert routing logits (b_ij) to softmax.

            for j in range(num_out):
                for i in range(num_in):
                    if s[j] == None:
                        s[j] = C[i,j]*u_hat[:,i,j,:] # [joint_batch, ]
                    else:
                        s[j] = C[i,j]*u_hat[:,i,j,:]

            v = [squash(s[j], dim=1) for j in range(num_out)]



            for i in range(num_in):
                for j in range(num_out):
                    u_vj1 = torch.dot(torch.mean(u_hat[:,i,j,:], dim=0), torch.mean(v[j], dim=0))
                    B[i,j] = B[i,j] + u_vj1



        return torch.cat(v,dim=2)