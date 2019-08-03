"""
    M. Mirza et al. Conditional Generative Adversarial Nets.
    https://arxiv.org/abs/1406.2661
"""

import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

from deepgen.gan.base.generator import GeneratorBase
from deepgen.gan.base.discriminator import DiscriminatorBase


class Generator(GeneratorBase):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]

            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def __str__(self):
        return 'Generator::' + str(self.model)


class Discriminator(DiscriminatorBase):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def __str__(self):
        return 'Discriminator::' + str(self.model)


class GAN:
    def __init__(self,
                 generator: nn.Module = None,
                 discriminator: nn.Module = None,
                 adversarial_loss: object = None,
                 optimizer_G: object = None,
                 optimizer_D: object = None,
                 cuda: bool = False):

        if generator is not None:
            self.generator = generator
        else:
            self.generator = Generator(latent_dim=100, img_shape=(1, 28, 28))

        if discriminator is not None:
            self.discriminator = discriminator
        else:
            self.discriminator = Discriminator(img_shape=(1, 28, 28))

        if adversarial_loss is not None:
            self.adversarial_loss = adversarial_loss
        else:
            self.adversarial_loss = nn.BCELoss()

        if optimizer_G is not None:
            self.optimizer_G = optimizer_G
        else:
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                                lr=0.0002,
                                                betas=(0.5, 0.999))

        if optimizer_D is not None:
            self.optimizer_D = optimizer_D
        else:
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                                lr=0.0002,
                                                betas=(0.5, 0.999))

        self.Tensor = torch.Tensor
        self.latent_dim = self.generator.latent_dim

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.Tensor = torch.cuda.FloatTensor

    def train(self,
              data_loader: DataLoader = None,
              n_epoch: int = 100,
              sample_interval = 10):

        for epoch in range(n_epoch):
            for i, (imgs, _) in enumerate(data_loader):
                # Adversarial ground truths
                valid = Variable(self.Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                real_imgs = Variable(imgs.type(self.Tensor))

                # Train generator
                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # Train discriminator
                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                print('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]'
                      % (epoch, n_epoch, i, len(data_loader), d_loss.item(), g_loss.item()))

                batches_done = epoch * len(data_loader) + i
                if batches_done % sample_interval == 0:
                    save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

    def __str__(self):
        return str(self.generator) + '\n' + str(self.discriminator)



