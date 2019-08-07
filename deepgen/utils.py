import torch

def total_params(model):
    total_parameters = sum(p.numel() for p in model.parameters())
    train_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'total parameters: ' + str(total_parameters) + '\n' \
           + 'train parameters: ' + str(train_parameters) + '\n'

def initW_normal_DCGAN(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(model.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(model.weight.data, 1.0, 0.2)
        torch.nn.init.constant_(model.bias.data, 0.0)
