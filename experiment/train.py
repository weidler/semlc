import torch
import torch.optim as optim
from torch.utils.data import DataLoader


def train(net, num_epoch, train_set, batch_size, criterion, learn_rate=0.01, check_loss=1000, optimizer=None,
          logger=None):
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    # Adam optimizer by default
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=learn_rate)
        # optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4)
        # optimizer = custom_optimizer_conv18(net)

    loss_history = []
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0

        if epoch >= 120 and epoch % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1

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

        if epoch % 5 == 0 or epoch == num_epoch - 1:
            logger.save_model(epoch)

        print('Finished Epoch')


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
