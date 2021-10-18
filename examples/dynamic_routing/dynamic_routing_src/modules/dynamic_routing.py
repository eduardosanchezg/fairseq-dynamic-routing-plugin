import torch
from torch import nn

from .utils import squash


class DynamicRouting(nn.Module):
    """ Dynamic routing procedure.
    Routing-by-agreement as in [1].
    Args:
        j (int): Number of parent capsules.
        n (int): Vector length of the parent capsules.
        bias_routing (bool): Add a bias parameter to the average parent predictions.
    """

    def __init__(self, j, n, bias_routing):
        super().__init__()
        self.soft_max = nn.Softmax(dim=1)
        self.j = j
        self.n = n

        # init depends on batch_size which depends on input size, declare dynamically in forward. see:
        # https://discuss.pytorch.org/t/dynamic-parameter-declaration-in-forward-function/427/2
        self.b_vec = None

        # init bias parameter
        if bias_routing:
            b_routing = nn.Parameter(torch.zeros(j, n))
            b_routing.data.fill_(0.1)
            self.bias = b_routing
        else:
            self.bias = None

        # log function that is called in the forward pass to enable analysis at end of each routing iter
        self.log_function = None

    def forward(self, u_hat, iters):
        """ Forward pass
        Args:
            u_hat (FloatTensor): Prediction vectors of the child capsules for the parent capsules. Shape: [batch_size,
                num parent caps, num child caps, len final caps]
            iters (int): Number of routing iterations.
        Returns:
            v_vec (FloatTensor): Tensor containing the squashed average predictions using the routing weights of the
                routing weight update. Shape: [batch_size, num parent capsules, len parent capsules]
        """

        b = u_hat.shape[0]  # batch_size
        i = u_hat.shape[2]  # number of parent capsules

        # init empty b_vec, on init would be better, but b and i are unknown there. Takes hardly any time this way.
        self.b_vec = torch.zeros(b, self.j, i, device="cuda", requires_grad=False)
        b_vec = self.b_vec

        # loop over all routing iterations
        for index in range(iters):

            # softmax over j, weight of all predictions should sum to 1
            c_vec = self.soft_max(b_vec)

            # created unsquashed prediction for parents capsules by a weighted sum over the child predictions
            # in einsum: bij, bjin-> bjn
            # in matmul: bj1i, bjin = bj (1i)(in) -> bjn
            s_vec = torch.matmul(c_vec.view(b, self.j, 1, i), u_hat.float()).squeeze()

            # add bias to s_vec
            if type(self.bias) == nn.Parameter:
                s_vec_bias = s_vec + self.bias

                # don't add a bias to capsules that have no activation add all
                # check which capsules where zero
                reset_mask = (s_vec.sum(dim=2) == 0)

                # set them back to zero again
                s_vec_bias[reset_mask, :] = 0
            else:
                s_vec_bias = s_vec

            # squash the average predictions
            v_vec = squash(s_vec_bias)

            # skip update last iter
            if index < (iters - 1):

                # compute the routing logit update
                # in einsum: "bjin, bjn-> bij", inner product over n
                # in matmul: bji1n, bj1n1 = bji (1n)(n1) = bji1
                b_vec_update = torch.matmul(u_hat.view(b, self.j, i, 1, self.n),
                                            v_vec.view(b, self.j, 1, self.n, 1)).view(b, self.j, i)

                # update b_vec
                # use x=x+1 instead of x+=1 to ensure new object creation and avoid inplace operation
                b_vec = b_vec + b_vec_update

            # call log function every routing iter for optional analysis
            if self.log_function:
                self.log_function(index, u_hat, b_vec, c_vec, v_vec, s_vec, s_vec_bias)

        return v_vec