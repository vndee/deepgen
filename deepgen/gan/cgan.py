"""
    M. Mirza et al. Conditional Generative Adversarial Nets.
    https://arxiv.org/abs/1406.2661
"""

import os
import time
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

from deepgen.gan.base import GeneratorBase
from deepgen.gan.base import DiscriminatorBase


class Generator(GeneratorBase):
    def __init__(self, latent_dim, img_shape, n_classes):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]

            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z, label):
        inp = torch.cat((self.label_emb(label), z), -1)
        img = self.model(inp)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def __str__(self):
        return 'Generator::' + str(self.model)


class Discriminator(DiscriminatorBase):
    def __init__(self, img_shape, n_classes):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape
        self.n_classes = n_classes
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(self.n_classes + int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, label):
        inp = torch.cat((img.view(img.size(0), -1), self.label_emb(label)), -1)
        return self.model(inp)

    def __str__(self):
        return 'Discriminator::' + str(self.model)


class CGAN:
    def __init__(self,
                 generator: nn.Module = None,
                 discriminator: nn.Module = None,
                 n_classes: int = 10,
                 adversarial_loss: object = None,
                 optimizer_G: object = None,
                 optimizer_D: object = None,
                 cuda: bool = False):

        if generator is not None:
            self.generator = generator
        else:
            self.generator = Generator(latent_dim=100, img_shape=(1, 32, 32), n_classes=n_classes)

        if discriminator is not None:
            self.discriminator = discriminator
        else:
            self.discriminator = Discriminator(img_shape=(1, 32, 32), n_classes=n_classes)

        if adversarial_loss is not None:
            self.adversarial_loss = adversarial_loss
        else:
            self.adversarial_loss = nn.MSELoss()

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

        self.FloatTensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor
        self.latent_dim = self.generator.latent_dim
        self.n_classes = n_classes

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.FloatTensor = torch.cuda.FloatTensor
            self.LongTensor = torch.cuda.LongTensor

    def sample_image(self, n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(self.FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(self.LongTensor(labels))
        gen_imgs = self.generator(z, labels)
        save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

    def train(self,
              data_loader: DataLoader = None,
              n_epoch: int = 100,
              sample_interval = 10,
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

            for i, (imgs, labels) in enumerate(data_loader):
                t0 = time.time()
                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(self.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                real_imgs = Variable(imgs.type(self.FloatTensor))
                labels = Variable(labels.type(self.LongTensor))

                # Train generator
                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                gen_labels = Variable(self.LongTensor(np.random.randint(0, self.n_classes, batch_size)))

                # Generate a batch of images
                gen_imgs = self.generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs, gen_labels), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # Train discriminator
                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(real_imgs, labels), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), gen_labels), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()
                t1 = time.time()

                if verbose:
                    print('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]'
                          % (epoch, n_epoch, i, len(data_loader), d_loss.item(), g_loss.item()))

                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                total_time += (t1 - t0)

                batches_done = epoch * len(data_loader) + i
                if batches_done % sample_interval == 0:
                    self.sample_image(n_row=10, batches_done=batches_done)
                    # save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

            training_history.append([total_g_loss/len(data_loader), total_d_loss/len(data_loader), total_time])
            if save_model_per_epoch > 0 and epoch % save_model_per_epoch == 0:
                self.generator.save(os.path.join('checkpoints', 'generator_' + str(epoch) + '_%.3f' % (total_g_loss/len(data_loader)) + '.dg'))
                self.discriminator.save(os.path.join('checkpoints', 'discriminator_' + str(epoch) + '_%.3f' % (total_d_loss/len(data_loader)) + '.dg'))

        return {'g_loss':   [training_history[_][0] for _ in range(len(training_history))],
                'd_loss':   [training_history[_][1] for _ in range(len(training_history))],
                'time':     [training_history[_][2] for _ in range(len(training_history))]}

    def __str__(self):
        return str(self.generator) + '\n' + str(self.discriminator)



