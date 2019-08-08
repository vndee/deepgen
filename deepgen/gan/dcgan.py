"""
    Alec Radford et al. Unsupervised representation learning with
    Deep Convolutional Generative Adversarial Networks.
    https://arxiv.org/pdf/1511.06434.pdf
"""

import os
import torch
import time
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

from deepgen.utils import total_params, initW_normal_DCGAN
from deepgen.gan.base import GANBase
from deepgen.gan.base import GeneratorBase
from deepgen.gan.base import DiscriminatorBase


class Generator(GeneratorBase):
    def __init__(self, latent_dim=100, img_size=32, channels=1):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.latent_dim = latent_dim

        self.model = nn.Sequential (
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        y = self.l1(z)
        y = y.view(y.shape[0], 128, self.init_size, self.init_size)
        return self.model(y)

    def infer(self, z):
        self.model.eval()
        return self.model(z)

    def __str__(self):
        return 'Generator::' + str(self.model)


class Discriminator(DiscriminatorBase):
    def __init__(self, img_size=32, channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.5)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        y = self.model(img)
        return self.adv_layer(y.view(y.shape[0], -1))

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def __str__(self):
        return 'Discriminator::' + str(self.model)


class DCGAN(GANBase):
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
            self.generator = Generator(latent_dim=100, img_size=32, channels=1)

        if discriminator is not None:
            self.discriminator = discriminator
        else:
            self.discriminator = Discriminator(img_size=32, channels=1)

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

        self.generator.apply(initW_normal_DCGAN)
        self.discriminator.apply(initW_normal_DCGAN)

    def train(self,
              data_loader: DataLoader = None,
              n_epoch: int = 100,
              sample_interval: int = 40,
              save_model_per_epoch: int = 1,
              verbose: bool = True):

        training_history = []
        if save_model_per_epoch > 0:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')

        for epoch in range(n_epoch):
            total_g_loss = 0.0
            total_d_loss = 0.0
            total_time = 0.0

            for i, (imgs, _) in enumerate(data_loader):
                # Adversarial ground truths
                t0 = time.time()
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
                t1 = time.time()

                if verbose:
                    print('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %.3f]'
                          % (epoch, n_epoch, i, len(data_loader), d_loss.item(), g_loss.item(), (t1 - t0)))

                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                total_time += (t1 - t0)

                batches_done = epoch * len(data_loader) + i
                if sample_interval > 0 and batches_done % sample_interval == 0:
                    save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

            training_history.append([total_g_loss/len(data_loader), total_d_loss/len(data_loader), total_time])
            if save_model_per_epoch > 0 and epoch % save_model_per_epoch == 0:
                self.generator.save(os.path.join('checkpoints', 'generator_' + str(epoch) + '_%.3f' % (total_g_loss/len(data_loader)) + '.dg'))
                self.discriminator.save(os.path.join('checkpoints', 'discriminator_' + str(epoch) + '_%.3f' % (total_d_loss/len(data_loader)) + '.dg'))

        return {'g_loss':   [training_history[_][0] for _ in range(len(training_history))],
                'd_loss':   [training_history[_][1] for _ in range(len(training_history))],
                'time':     [training_history[_][2] for _ in range(len(training_history))]}

    def __str__(self):
        return str(self.generator) + '\n'\
               + total_params(self.generator) + '\n'\
               + str(self.discriminator) + '\n' \
               + total_params(self.discriminator)
