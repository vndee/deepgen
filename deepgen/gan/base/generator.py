import torch.nn as nn


class GeneratorBase(nn.Module):
    def __init__(self):
        super(GeneratorBase, self).__init__()

    def forward(self, *input):
        return NotImplementedError

    def __str__(self):
        return NotImplementedError

    def save(self):
        return None

    def load(self):
        return None