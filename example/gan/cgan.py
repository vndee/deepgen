import os
import torch

from deepgen.gan.cgan import CGAN
from torchvision import datasets
import torchvision.transforms as transforms

img_size = (1, 28, 28)
batch_size = 64

if __name__ == '__main__':
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data/',
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    os.makedirs('images', exist_ok=True)

    model = CGAN()

    print(model)
    his = model.train(data_loader=data_loader, n_epoch=2, sample_interval=400)
    print(his)
