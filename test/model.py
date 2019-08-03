import unittest
import numpy as np
import torch
from torch.autograd import Variable
from deepgen.gan.bgan import Generator
from deepgen.gan.bgan import Discriminator


class TestModelManipulationMethods(unittest.TestCase):
    def test_load_generator(self):
        generator = Generator()
        generator.load('../example/gan/checkpoints/generator_0_1.290.dg')

    def test_load_discriminator(self):
        discriminator = Discriminator()
        discriminator.load('../example/gan/checkpoints/discriminator_0_0.328.dg')

    def test_generator_infer(self):
        z = Variable(torch.Tensor(np.random.normal(0, 1, (1, 100))))
        generator = Generator()
        generator.load('../example/gan/checkpoints/generator_0_1.290.dg')
        im = generator.infer(z)
        self.assertEqual(im.shape, (1, 28*28))

    def test_discriminator_infer(self):
        z = torch.rand([1, 784])
        discriminator = Discriminator()
        discriminator.load('../example/gan/checkpoints/discriminator_0_0.328.dg')
        val = discriminator.infer(z)
        self.assertEqual(val.shape, torch.Size([1, 1]))


if __name__ == '__main__':
    unittest.main()
