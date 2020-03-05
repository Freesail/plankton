from torchvision.models import *
from torch import nn


def get_mlp(in_dim, hiddent_dim, out_dim):
    layers = []
    hiddent_dim.insert(0, in_dim)
    hiddent_dim.append(out_dim)

    for i in range(len(hiddent_dim) - 1):
        layers.append(nn.Linear(hiddent_dim[i], hiddent_dim[i + 1]))
        layers.append(nn.ReLU())
    layers.pop()
    return nn.Sequential(*tuple(layers))


def get_model(backbone, fc_hidden_dim, num_classes, device):
    model_conv = eval('%s(pretrained=True)' % backbone)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = get_mlp(num_ftrs, fc_hidden_dim, num_classes)
    model_conv = model_conv.to(device)
    return model_conv
