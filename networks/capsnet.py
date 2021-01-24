"""An adaption of Adam Bielski's implementation (https://github.com/adambielski/CapsNet-pytorch/blob/master/net.py)"""
import math

import core

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import BaseNetwork
from layers.base import BaseSemLCLayer
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler


def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)), requires_grad=True)

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b, dim=1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim), requires_grad=True)
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class CapsNet(BaseNetwork):
    def __init__(self, input_shape, n_classes: int, routing_iterations=3, lateral_layer: BaseSemLCLayer = None,
                 complex_cells: bool = False):
        super(CapsNet, self).__init__(input_shape=input_shape, lateral_layer=lateral_layer)

        self.conv_one = nn.Conv2d(self.input_channels, 256, kernel_size=9, stride=1)
        conv_one_out_size = self.conv_one(self.generate_random_input()).shape

        if self.lateral_layer_partial is not None:
            self.lateral_layer = self.lateral_layer_partial(self.conv_one,
                                                            ricker_width=(self.conv_one.out_channels / 64) * 3)
            self.lateral_layer.compile(conv_one_out_size[-2:])

        self.primaryCaps = PrimaryCapsLayer(conv_one_out_size[-3], 32, 8, kernel_size=9, stride=2)  # outputs 6*6

        self.num_primaryCaps = 32 * (((self.input_height - 8) - 8) // 2) * (((self.input_width - 8) - 8) // 2)
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)

    def forward(self, x):
        out_conv_one = self.conv_one(x)

        if self.lateral_layer_partial is not None:
            out_conv_one = self.lateral_layer(out_conv_one)

        primary_capsules = self.primaryCaps(F.relu(out_conv_one))
        digit_capsules = self.digitCaps(primary_capsules)
        probs = digit_capsules.pow(2).sum(dim=2).sqrt()

        return probs

    @staticmethod
    def make_preferred_criterion():
        return MarginLoss(0.9, 0.1, 0.5)

    def make_preferred_optimizer(self):
        return optim.Adam(self.parameters(), lr=0.001)

    @staticmethod
    def make_preferred_lr_schedule(optimizer):
        return lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()