import random
import time

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim

from model.network.benchmark import BaseDenoisingCNN
from transforms.noise import GaussianNoise
from util import cifar

if __name__ == "__main__":
    torch.random.manual_seed(1000)
    random.seed(1000)

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    print(f"USE CUDA: {use_cuda}.")

    transformer = torchvision.transforms.ToTensor()
    pil_transformer = torchvision.transforms.ToPILImage()
    noise_transformer = GaussianNoise()
    train = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True,
                                         transform=transformer)
    train_noisy = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True,
                                               transform=torchvision.transforms.Compose(
                                                   [transformer, noise_transformer]
                                               ))
    test = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True,
                                        transform=transformer)
    test_noisy = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True,
                                              transform=torchvision.transforms.Compose(
                                                  [transformer, noise_transformer]
                                              ))

    label_names = cifar.unpickle("../data/cifar10/cifar-10-batches-py/batches.meta")[b"label_names"]

    autoencoder = BaseDenoisingCNN()
    optimizer = optim.SGD(autoencoder.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    epochs = 10
    print_every = 10000
    thin = 0 #49900

    # TRAIN
    sample_order = list(range(len(train) - thin))
    images_seen = 0
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"------- Epoch {epoch} -------")
        accumulated_loss = 0
        random.shuffle(sample_order)
        for sample_id in sample_order:
            optimizer.zero_grad()

            image_tensor, _ = train[sample_id]
            noisy_tensor, _ = train_noisy[sample_id]
            image_tensor.unsqueeze_(0)
            noisy_tensor.unsqueeze_(0)
            if use_cuda:
                image_tensor = image_tensor.cuda()
                noisy_tensor = noisy_tensor.cuda()

            denoised_tensor = autoencoder(noisy_tensor)
            loss = criterion(denoised_tensor, image_tensor)
            loss.backward()
            accumulated_loss += loss.item()

            optimizer.step()
            images_seen += 1

            if images_seen % print_every == 0:
                time_passed = time.time() - start_time
                start_time = time.time()
                print(f"{images_seen}/{epochs*len(sample_order)}: {round(accumulated_loss/print_every, 4)} "
                      f"(took {round(time_passed, 2)}s).")
                accumulated_loss = 0

    # TEST
    i = 0
    for i in range(len(test)):
        image_tensor, label = test[i]
        noisy_tensor, _ = test_noisy[i]

        image_tensor.unsqueeze_(0)
        noisy_tensor.unsqueeze_(0)

        if use_cuda:
            image_tensor = image_tensor.cuda()
            noisy_tensor = noisy_tensor.cuda()

        denoised_tensor = autoencoder(noisy_tensor)
        loss = criterion(denoised_tensor, image_tensor)

        fig: plt.Figure
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(10, 4)
        fig.suptitle(f"A {str(label_names[label])} reconstructed with loss {round(loss.item(), 5)}")

        axs[0].imshow(pil_transformer(image_tensor.cpu().squeeze_()))
        axs[0].set_title("Original")
        axs[1].imshow(pil_transformer(noisy_tensor))
        axs[1].set_title("Noisy")
        axs[2].imshow(pil_transformer(denoised_tensor.cpu().squeeze_()))
        axs[2].set_title("Reconstruction")
        plt.show()

        i += 1
