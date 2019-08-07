import os
import torch

from deepgen.gan.dcgan import DCGAN
from torchvision import datasets
import torchvision.transforms as transforms

img_size = (1, 28, 28)
batch_size = 64

if __name__ == '__main__':
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data/',
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    os.makedirs('images', exist_ok=True)

    model = DCGAN()

    print(model)
    his = model.train(data_loader=data_loader, n_epoch=5, sample_interval=40)
    print(his)
