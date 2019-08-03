## DeepGen
A collection of modern Deep Generative Models.

### Install

`pip3 install deepgen`

### Usage

Toy example:
```python
import os
import torch

from deepgen.gan.gan import GAN
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
                [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    os.makedirs('images', exist_ok=True)

    model = GAN()

    print(model)
    his = model.train(data_loader=data_loader, n_epoch=5, sample_interval=10)
    print(his)

```

### TODO:

#### Adversarial Generative Network (GANs)

- [x] Vanilla Adversarial Generative Neural Network
- [ ] Boundary Seeking Adversarial Generative Neural Network

#### Variational Autoencoder (VAEs)

- [ ] Vanilla Variational Autoencoder
