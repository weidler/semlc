import torch.optim as optim
from torch.utils.data import DataLoader


def train(net, num_epoch, train_set, batch_size, criterion, learn_rate=0.01, check_loss=1000, logger=None):
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    # Adam optimizer by default
    # optimizer = optim.Adam(net.parameters(), lr=learn_rate)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loss_history = []
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
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

        print('Finished Training')


