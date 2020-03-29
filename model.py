from torchvision.models import *
from torch import nn
import torch
import copy
from data import safe_listdir
from torchvision import transforms, datasets
import os
from torch.utils.data.dataset import ConcatDataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np


def helper_mlp(in_dim, hidden_dim, out_dim, fc_dropout, is_bayes=False):
    if is_bayes:
        return BayesianMlp(in_dim, hidden_dim, out_dim, fc_dropout)
    else:
        layers = []
        hidden_dim = copy.deepcopy(hidden_dim)
        hidden_dim.insert(0, in_dim)
        hidden_dim.append(out_dim)

        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())
            if fc_dropout > 0:
                layers.append(nn.Dropout(p=fc_dropout))
        layers.pop()
        layers.pop()
        return nn.Sequential(*tuple(layers))


class BayesianMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, fc_dropout):
        super(BayesianMlp, self).__init__()
        self.p_drop = fc_dropout
        hidden_dim = copy.deepcopy(hidden_dim)
        hidden_dim.insert(0, in_dim)
        hidden_dim.append(out_dim)
        self.num_layers = len(hidden_dim) - 1
        for i in range(self.num_layers):
            setattr(self, 'linear_layer{}'.format(i),
                    nn.Linear(hidden_dim[i], hidden_dim[i + 1]))

    def forward(self, x, n_samples=100):
        if self.training:
            for i in range(self.num_layers):
                x = getattr(self, 'linear_layer{}'.format(i))(x)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.p_drop)
            return x
        else:
            preds = []
            self.train()
            for i in range(n_samples):
                preds.append(self.forward(x))
            pred = torch.stack(preds, dim=0).mean(dim=0)
            self.eval()
            return pred
