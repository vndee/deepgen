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

Updating

#### Generative Adversarial Network (GANs)

- [x] Vanilla GAN
- [x] Boundary Seeking GAN
- [ ] Auxiliary Classifier GAN
- [ ] BEGAN
- [ ] BicycleGAN
- [ ] Boundary-Seeking GAN
- [ ] Conditional GAN
- [ ] Context-Conditional GAN
- [ ] Context Encoder
- [ ] Coupled GAN
- [ ] CycleGAN
- [ ] Deep Convolutional GAN
- [ ] DiscoGAN
- [ ] DRAGAN
- [ ] DualGAN
- [ ] Energy-Based GAN
- [ ] Enhanced Super-Resolution GAN
- [ ] GAN
- [ ] InfoGAN
- [ ] Least Squares GAN
- [ ] MUNIT
- [ ] Pix2Pix
- [ ] PixelDA
- [ ] Relativistic GAN
- [ ] Semi-Supervised GAN
- [ ] Softmax GAN
- [ ] StarGAN
- [ ] Super-Resolution GAN
- [ ] UNIT
- [ ] Wasserstein GAN
- [ ] Wasserstein GAN GP
- [ ] Wasserstein GAN DIV

#### Variational Autoencoder (VAEs)

- [ ] Vanilla VAE
- [ ] Conditional VAE
- [ ] Denoising VAE
- [ ] Adversarial Variational Bayes
- [ ] Adversarial Autoencoder
