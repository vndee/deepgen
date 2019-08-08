import torch
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

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))


class DiscriminatorBase(nn.Module):
    def __init__(self):
        super(DiscriminatorBase, self).__init__()

    def forward(self, *input):
        return NotImplementedError

    def __str__(self):
        return NotImplementedError

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))


