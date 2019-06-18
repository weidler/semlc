import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from util.ourlogging import Logger
from model.network.alexnet_paper import ConvNet11, ConvNet18

def train(net_name, num_epoch, train_loader, val_loader, criterion, use_cuda, learn_rate=0.01, check_loss=1000, custom_optimizer=True, logger=None, path='best_model.pt'):
    # hyperparams for SmallAlexnetInhibition
    # scope is specific to each layer
    # Conv11
    assert net_name in ['conv11', 'conv18']
    if net_name == 'conv11':
        range_scope = [[33, 35, 37, 39], [33, 35, 37, 39], [33, 35, 37, 39], [17, 19, 21, 23]]
    # Conv18
    elif net_name == 'conv18':
        range_scope = [[17, 19, 21, 23], [17, 19, 21, 23], [33, 35, 37, 39]]
    range_ricker_width = np.linspace(3.0, 6.0, num=4)
    range_damp = np.linspace(0.1, 0.16, num=4)
    best_loss = np.Infinity
    for scope in range_scope:
        for ricker_width in range_ricker_width:
            for damp in range_damp:
                if net_name == 'conv11':
                    net = ConvNet11(scope, ricker_width, damp)
                elif net_name == 'conv18':
                    net = ConvNet18(scope, ricker_width, damp)
                # Adam optimizer by default
                if not custom_optimizer:
                    optimizer = optim.Adam(net.parameters(), lr=learn_rate)
                    # optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4)
                    # optimizer = custom_optimizer_conv18(net)
                else:
                    optimizer = custom_optimizer_conv18(net)
                if use_cuda:
                    net.cuda()
                logger = Logger(net)
                loss_history = []
                for epoch in range(num_epoch):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, (inputs, labels) in enumerate(train_loader, 0):
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        if torch.cuda.is_available():
                            outputs = net(inputs.cuda())
                        else:
                            outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # print statistics
                        running_loss += loss.item()
                        if i % check_loss == check_loss - 1:
                            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / check_loss))
                            loss_history.append(running_loss / check_loss)
                            if logger is not None:
                                logger.update(running_loss / check_loss, epoch)
                            running_loss = 0.0

                    print('Finished Epoch')

                if epoch % 5 == 0 or epoch == num_epoch - 1:
                    logger.save_model(epoch)
                # check validation loss to decide which model we keep
                for i, (inputs, labels) in enumerate(val_loader, 0):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    if torch.cuda.is_available():
                        outputs = net(inputs.cuda())
                    else:
                        outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                if best_loss > running_loss:
                    best_loss = running_loss
                    # save the model
                    torch.save(net.state_dict(), path)
                    best_net = net
    return best_net


def custom_optimizer_conv18(model):
    # optim.SGD()
    optimizer = optim.SGD(
        [
            {"params": model.conv1.weight},
            {"params": model.conv2.weight},
            {"params": model.conv3.weight},
            {"params": model.fc.weight, "weight_decay": 1},
            {"params": model.conv1.bias, "lr": 2e-3},
            {"params": model.conv2.bias, "lr": 2e-3},
            {"params": model.conv3.bias, "lr": 2e-3},
            {"params": model.fc.bias, "lr": 2e-3, "weight_decay": 1},
        ],
        lr=1e-3, momentum=0.9, weight_decay=4e-3
    )
    return optimizer


def custom_optimizer_conv11(model):
    # optim.SGD()
    optimizer = optim.SGD(
        [
            {"params": model.conv1.weight, "weight_decay": 0},
            {"params": model.conv2.weight, "weight_decay": 0},
            {"params": model.conv3.weight, "weight_decay": 4e-3},
            {"params": model.conv4.weight, "weight_decay": 4e-3},
            {"params": model.fc.weight, "weight_decay": 1e-2},
            {"params": model.conv1.bias, "lr": 2e-3, "weight_decay": 0},
            {"params": model.conv2.bias, "lr": 2e-3, "weight_decay": 0},
            {"params": model.conv3.bias, "lr": 2e-3, "weight_decay": 4e-3},
            {"params": model.conv4.bias, "lr": 2e-3, "weight_decay": 4e-3},
            {"params": model.fc.bias, "lr": 2e-3, "weight_decay": 1e-2},
        ],
        lr=1e-3, momentum=0.9
    )
    return optimizer


if __name__ == "__main__":
    from model.network.alexnet_paper import ConvNet18, ConvNet11
    model = ConvNet11()
    model2 = ConvNet18()
    opt = custom_optimizer_conv11(model)
    print(opt)
    opt = custom_optimizer_conv18(model2)
    print(opt)
