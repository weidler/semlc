import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.eval import accuracy_from_data_loader


def train_model(net, num_epoch, train_loader, criterion, learn_rate=0.01, val_loader=None, optimizer=None,
                logger=None, verbose=False, save_freq=40):
    # Adam optimizer by default
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=learn_rate)

    loss_history = []
    max_val_acc = 0
    num_batches = train_loader.__len__()
    for epoch in tqdm(range(num_epoch), disable=verbose):  # loop over the dataset multiple times
        running_loss = 0.0
        if epoch == 100:
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

            if i == num_batches - 1:
                log_loss = running_loss / num_batches
                loss_history.append(log_loss)
                if logger is not None:
                    if val_loader is None:
                        logger.update_loss(log_loss, epoch + 1)
                        logger.log('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, log_loss), console=verbose)
                    else:
                        val_acc = accuracy_from_data_loader(net, val_loader)
                        if val_acc > max_val_acc:
                            max_val_acc = val_acc
                            if epoch >= 100:
                                logger.save_model(f'{epoch + 1}_best', best=True)
                        logger.log('[%d, %5d] loss: %.3f val_acc: %.3f' % (epoch + 1, i + 1, log_loss, val_acc),
                                   console=verbose)
                    logger.update_loss(log_loss, epoch + 1)
                running_loss = 0.0

        if epoch > 0 and epoch % save_freq == 0:
            logger.save_model(epoch + 1)


def train(net, num_epoch, train_set, batch_size, criterion, learn_rate=0.01, val_set=None, optimizer=None, logger=None,
          verbose=False, save_freq=40):
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set,
                                batch_size=batch_size,
                                shuffle=False)

    train_model(net, num_epoch, train_loader, criterion,
                learn_rate=learn_rate,
                val_loader=val_loader,
                optimizer=optimizer,
                logger=logger,
                verbose=verbose,
                save_freq=save_freq)


if __name__ == "__main__":
    from model.network.alexnet_paper import InhibitionNetwork

    model = InhibitionNetwork()
