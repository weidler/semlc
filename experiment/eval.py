import torch
from torch.utils.data import DataLoader


def accuracy(net, data_set, batch_size):
    data_loader = DataLoader(data_set, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total