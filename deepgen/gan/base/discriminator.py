import torch.nn as nn


class DiscriminatorBase(nn.Module):
    def __init__(self):
        super(DiscriminatorBase, self).__init__()

    def forward(self, *input):
        return NotImplementedError

    def __str__(self):
        return NotImplementedError

    def save(self):
        return None

    def load(self):
        return None