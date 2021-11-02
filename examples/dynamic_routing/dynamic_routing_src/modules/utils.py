
"""Utilities
PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829
Author: Cedric Chee
"""
import os.path

import numpy as np
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





def squash(sj):
    """
    The non-linear activation used in Capsule.
    It drives the length of a large vector to near 1 and small vector to 0
    This implement equation 1 from the paper.
    """

    #sj[sj == 0.0] = -0.001

    #if not torch.isfinite(sj).all():
    # if (sj == 0.0).sum().item != 0:
    #     print("THERE'S SOMETHING WRONG WITH THE INPUT")
    #     for i in range(sj.size(0)):
    #         if (sj[i,:] == 0.0).sum().item != 0:
    #                 print(i)
    #                 print(sj[i,:])

    norm = torch.linalg.vector_norm(sj, dim=1)

    if (norm == 0.0).sum().item() != 0:
        print("THERE'S SOMETHING WRONG WITH THE NORM")
        for i in range(norm.size(0)):
            if (norm[i] == 0.0).item():
                print(i)
                print(norm[i])

    norm = torch.stack([norm]*sj.size(1),dim=1)
    sq_norm = norm * norm

    # if torch.isnan(sq_norm).any():
    #     print("||||||||||||SQ NORM|||||||||||||")
    #     print(sq_norm)

    num = sq_norm * sj  #   sjq_norm[bsz,output_dim] sj[bsz,output_dim]

    # if torch.isnan(num).any():
    #     print("|||||||||||| NUM |||||||||||||")
    #     print(num)

    den = (1 + sq_norm) * norm

    # if torch.isnan(den).any():
    #     print("|||||||||||| DEN |||||||||||||")
    #     print(den)

    vj = torch.div(num, den)
    if not torch.isfinite(vj).all():
        break_out = False
        for i in range(sj.size(0)):
            for j in range(sj.size(1)):
                a = num[i,j] / den[i,j]
                if not torch.isfinite(a).item():
                    print("||||||a||||||||||||||")
                    print(i)
                    print(j)
                    print(a)
                    print(num[i,:])
                    print(den[i,:])
                    print(norm[i,:])
                    print(sj[i,:])
                    # if not os.path.exists("norm.pt"):
                    #     torch.save(norm,"norm.pt")
                    #     torch.save(num,"num.pt")
                    #     torch.save(den,"den.pt")
                    #     torch.save(sj,"sj.pt")
                    #     torch.save(vj,"vj.pt")

                    break
                    break_out = True
            if break_out:
                break



    if not torch.isfinite(vj).all():
        print("||||||||||||V_j|||||||||||||")
        print(vj)

    return vj


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


