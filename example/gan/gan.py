import os
import torch

from deepgen.gan.gan import GAN
from deepgen.gan.gan import Generator
from deepgen.gan.gan import Discriminator

from torchvision import datasets
import torchvision.transforms as transforms

img_size = (1, 28, 28)
batch_size = 64

if __name__ == '__main__':
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data/',
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    os.makedirs('images', exist_ok=True)

    # G = Generator(latent_dim=100, img_shape=(1, 28, 28))
    # D = Discriminator(img_shape=(1, 28, 28))

    model = GAN()

    print(model)
    model.train(dataloader=dataloader, n_epoch=10, sample_interval=200)
