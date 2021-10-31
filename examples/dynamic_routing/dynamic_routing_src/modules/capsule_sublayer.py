import scipy
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
import scipy.special as scps


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

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
        B = np.zeros((num_in, num_out))



        # if self.cuda_enabled:
        #     B = B.cuda()
        num_iterations = self.num_routing

        for iteration in range(num_iterations):
            # Routing algorithm

            C = scipy.special.softmax(B, axis= 0) # Convert routing logits (b_ij) to softmax.

            if str(np.mean(np.absolute(C))) == "nan":
                print("|||||||||||C||||||||||||||")
                print(C)
                print(iteration)

            for j in range(num_out):
                for i in range(num_in):
                    if s[j] == None:
                        s[j] = C[i,j]*u_hat[:,i,j,:] # [joint_batch, ]
                    else:
                        s[j] = C[i,j]*u_hat[:,i,j,:]
                    if str(s[j].abs().mean().item()) == "nan":
                        print("|||||||||||s_j||||||||||||||")
                        print(s[j])
                        print(i)
                        print(j)
                        print(iteration)
                        break


            v = [squash(s[j], dim=1) for j in range(num_out)]

            for i in range(len(v)):
                if str(s[j].abs().mean().item()) == "nan":
                    print("|||||||||||||||||V||||||||||||")
                    print(v[i])
                    print(i)

            for i in range(num_in):
                for j in range(num_out):
                    u_vj1 = torch.dot(torch.mean(u_hat[:,i,j,:], dim=0), torch.mean(v[j], dim=0))
                    if str(np.mean(np.absolute(B))) == "nan":
                        print("|||||||||||B||||||||||||||")
                        print(B)
                        print(i)
                        print(j)
                        print(iteration)
                        print(">>>>>>>>>>>>u_vj1")
                        print(u_vj1)
                        print(">>>>>>>>>>>>>u_hat")
                        print(torch.mean(u_hat[:, i, j, :], dim=0))
                        print(">>>>>>>>>>>>>>>v_j")
                        print(v[j])
                        print(">>>>>>>>>>>>>>>v_j mean")
                        print(torch.mean(v[j], dim=0))
                    B[i,j] = B[i,j] + u_vj1

            if str(np.mean(np.absolute(B))) == "nan":
                print("|||||||||||B||||||||||||||")
                print(B)
                print("--end of b loop--")
                print(iteration)
                break

        print("||||||||||||||absmean||||||||||||")
        print("w: " + str(self.weights.abs().mean()) + " v[0]: " + str(v[0].abs().mean()) + " B: " + str(np.mean(np.absolute(B))))
        if str(np.mean(np.absolute(B))) == "nan":
            print("|||||||||||||||||||||||||||u_hat|||||||||||||||||||||||")
            print(u_hat)
            print("||||||||||||||||||||||||u")
            print(u)
            print("|||||||||||||||||||||||||||s_0||||||||||||||||||||||||||")
            print(s[0])
            print("|||||||||||||||||||||||||||v_0||||||||||||||||||||||||||||")
            print(v[0])
            print("||||||||||||||||||||||||||||||B||||||||||||||||||||||||||||")
            print(B)

        return torch.stack(v,dim=2).permute(2,0,1).reshape(num_out,bsz,seq_len,out_dim)