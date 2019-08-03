import torch.nn as nn


class GANBase:
    def train(self) -> dict:
        return NotImplementedError

    def __str__(self):
        return NotImplementedError


class GeneratorBase(nn.Module):
    def __init__(self):
        super(GeneratorBase, self).__init__()

    def forward(self, *input):
        return NotImplementedError

    def __str__(self):
        return NotImplementedError

    def infer(self, *args, **kwargs):
        return NotImplementedError

    def save(self):
        return None

    def load(self):
        return None


class DiscriminatorBase(nn.Module):
    def __init__(self):
        super(DiscriminatorBase, self).__init__()

    def forward(self, *input):
        return NotImplementedError

    def __str__(self):
        return NotImplementedError

    def infer(self, *args, **kwargs):
        return NotImplementedError

    def save(self):
        return None

    def load(self):
        return None

