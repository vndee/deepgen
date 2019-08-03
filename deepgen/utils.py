def total_params(model):
    total_parameters = sum(p.numel() for p in model.parameters())
    train_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'total parameters: ' + str(total_parameters) + '\n' \
           + 'train parameters: ' + str(train_parameters) + '\n'