"""A copy of Adam Bielski's implementation (https://github.com/adambielski/CapsNet-pytorch/blob/master/net.py),
only adapted to include our inhibition."""
import sys

from networks.base import BaseNetwork

sys.path.append("/")

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torch.utils.data import Subset

from util.eval import accuracies_from_list
from util.log import ExperimentLogger


def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
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
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
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


class CapsNet(BaseNetwork, nn.Module):

    def __init__(self, routing_iterations, n_classes=10):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=9, stride=2)  # outputs 6*6
        self.num_primaryCaps = 32 * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


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


# CAPS NET WITH OPTION FOR LC

class InhibitionCapsNet(BaseNetwork, nn.Module):
    def __init__(self, widths: List[int], damps: List[float], strategy: str, optim: str, n_classes: int = 10):
        super().__init__(widths, damps, strategy, optim)

        # check if legal parameters
        assert len(widths) == 1, "Cannot have more than one LC layers in CapsNet, because there is only one conv layers."

        # primary convolution
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)

        # inhibition layers
        self.inhibition_layer = self.lateral_connect_layer_type(num_layer=1, in_channels=256)

        # primary capsules
        self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=9, stride=2)  # outputs 6*6
        self.num_primaryCaps = 32 * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, 3)

        # output class capsules
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)

    def forward(self, input):
        x = self.conv1(input)
        x = self.inhibition_layer(x)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()

        return x, probs


if __name__ == '__main__':
    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable

    # Training settings
    parser = argparse.ArgumentParser(description='CapsNet with MNIST')
    parser.add_argument("strategy", type=str, choices=['baseline', 'lc'])
    parser.add_argument("-i", type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--no-inhibition", action="store_true", default=False, help="disables Inhibition layers")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--routing_iterations', type=int, default=3)
    parser.add_argument('--with_reconstruction', action='store_true', default=False)
    parser.add_argument('--save_freq', type=int, default=50)

    # lc parameters
    parser.add_argument('--scope', type=int, default=255, help='scope of the connectivity profile wavelet')
    parser.add_argument('--width', type=float, default=12.0, help='width of the connectivity profile wavelet')
    parser.add_argument('--damp', type=float, default=0.2, help='damping of the connectivity profile wavelet')
    parser.add_argument('--lc-strat', type=str, default="CLC", help='connectivity strategy')
    parser.add_argument('--optim', type=str, default="frozen", help='optimization of the profile')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        # Subset(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(2), transforms.RandomCrop(28),
                           transforms.ToTensor()
                       ]))
        #    , indices=[i for i in range(100)])
        , batch_size=args.batch_size, shuffle=True, **kwargs)

    test_set = datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ]))

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # CUDA
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(f"USE CUDA: YES.")


    def train(epoch, logger):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                data, target = Variable(data), Variable(target)

            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784), size_average=False).item()
                test_loss += loss_fn(probs, target, size_average=False).item()
                test_loss += reconstruction_alpha * reconstruction_loss
            else:
                output, probs = model(data)
                test_loss += loss_fn(probs, target, size_average=False).item()

            pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        val_acc = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            val_acc))
        return test_loss, val_acc


    networks = []
    accuracies = []

    for i in range(0, args.i):
        if args.strategy == 'baseline':
            model = CapsNet()
        elif args.strategy == 'lc':
            model = InhibitionCapsNet(widths=[args.width], damps=[args.damp], strategy=args.lc_strat, optim=args.optim)

        if args.with_reconstruction:
            reconstruction_model = ReconstructionNet(16, 10)
            reconstruction_alpha = 0.0005
            model = CapsNetWithReconstruction(model, reconstruction_model)

        if args.cuda:
            model.cuda()

        logger = Logger(model, experiment_code=f"{args.strategy}_{i}")

        # layers.load_state_dict(torch.load('output/15802478382228594699_best.layers', map_location=lambda storage, loc: storage))

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer.load_state_dict(torch.load('./output/15802478382228594699_best.opt', map_location=lambda storage, loc: storage))

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

        loss_fn = MarginLoss(0.9, 0.1, 0.5)

        networks.append(model)

        max_val_acc = 0

        for epoch in range(1, args.epochs + 1):
            train(epoch, logger)
            test_loss, val_acc = test()

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                logger.save_model(epoch + 1, best=True)
                logger.save_optimizer(optimizer, epoch + 1, best=True)

            logger.log('[%d, %5d] loss: %.3f val_acc: %.3f' % (epoch + 1, i + 1, test_loss, val_acc))

            scheduler.step(test_loss)
            torch.save(model.state_dict(),
                       './output/capsnet/{:02d}_{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(i, epoch,
                                                                                                         args.routing_iterations,
                                                                                                         args.with_reconstruction))
            if epoch > 0 and epoch % args.save_freq == 0:
                logger.save_model(epoch + 1)
                logger.save_optimizer(optimizer, epoch + 1)

        accuracies.append(max_val_acc)

    print(accuracies)
    acc = accuracies_from_list(accuracies)
    print(f"{acc}")
