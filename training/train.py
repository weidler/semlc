import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


def train(net, num_epoch, trainset, batch_size, criterion, learn_rate=0.01, check_loss=100):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    #Adam optimizer by default
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % check_loss == check_loss-1:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / check_loss))
                running_loss = 0.0
        print('Finished Training')

def accuracy(dataset, net, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total