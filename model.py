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


# yuan yuan is coming


class BayesianMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, fc_dropout):
        super(BayesianMlp, self).__init__()
        self.p_drop = fc_dropout
        self.linear_layers = self.get_linear_layers(in_dim, hidden_dim, out_dim)

    @staticmethod
    def get_linear_layers(in_dim, hidden_dim, out_dim):
        layers = []
        hidden_dim = copy.deepcopy(hidden_dim)
        hidden_dim.insert(0, in_dim)
        hidden_dim.append(out_dim)
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
        return layers

    def forward(self, x):
        num_layers = len(self.linear_layers)
        if self.training:
            for i in range(num_layers):
                x = self.linear_layers[i](x)
                if i < num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.p_drop)
            return x
        else:
            pass


def helper_mlp(in_dim, hidden_dim, out_dim, fc_dropout):
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


def helper_model(backbone, pretrained, fc_hidden_dim, fc_dropout, tune_conv, num_classes, device):
    if backbone == 'resnet18':
        model_conv = resnet18(pretrained=pretrained)
    elif backbone == 'wide_resnet50_2':
        model_conv = wide_resnet50_2(pretrained=pretrained)
    else:
        raise NotImplementedError

    if pretrained and (not tune_conv):
        for param in model_conv.parameters():
            param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = helper_mlp(num_ftrs, fc_hidden_dim, num_classes, fc_dropout)
    fc_params = model_conv.fc.parameters()
    model_conv = model_conv.to(device)

    ignored_params = list(map(id, model_conv.fc.parameters()))
    conv_params = filter(lambda p: id(p) not in ignored_params,
                         model_conv.parameters())
    return model_conv, conv_params, fc_params


def helper_train(model_cfg, optimizer_cfg, scheduler_cfg):
    model, conv_params, fc_params = helper_model(**model_cfg)
    if model_cfg['pretrained'] and (not model_cfg['tune_conv']):
        optimizer = torch.optim.Adam(model.fc.parameters(), **optimizer_cfg)
    else:
        optimizer = torch.optim.Adam(
            [
                {'params': conv_params,
                 'lr': optimizer_cfg['conv_lr_ratio'] * optimizer_cfg['lr']},
                {'params': fc_params,
                 'lr': optimizer_cfg['lr']}
            ],
        )
    scheduler = StepLR(optimizer, **scheduler_cfg)
    return model, optimizer, scheduler


def helper_dataloaders(image_datasets, batch_size):
    class_names = image_datasets['val'].classes
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=16)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    if 'unlabel' in image_datasets.keys():
        dataloaders['unlabel'] = torch.utils.data.DataLoader(image_datasets['unlabel'],
                                                             batch_size=int(batch_size / 4),
                                                             shuffle=True, num_workers=16)
        dataset_sizes['unlabel'] = 0

    return class_names, dataloaders, dataset_sizes


def helper_class_loss_acc(class_loss, class_acc, class_cnt, losses, preds, labels):
    labels = labels.cpu().numpy().astype(np.int)
    preds = preds.cpu().numpy().astype(np.int)
    losses = losses.cpu().numpy().astype(np.float)

    for i in range(labels.shape[0]):
        class_cnt[labels[i]] = class_cnt[labels[i]] + 1
        class_loss[labels[i]] = class_cnt[labels[i]] + losses[i]
        class_acc[labels[i]] = class_acc[labels[i]] + (preds[i] == labels[i])


def train_model(class_names, dataset_sizes,
                dataloaders,
                model, loss_fn, optimizer, scheduler, device,
                num_epochs=50, batch_per_disp=128, pseudo=False, pseudo_para=0):
    result = \
        {
            'best_model': None,
            'best_loss': 100,
            'best_acc': 0,
            'val_class_cnt': None,
            'best_class_loss': None,
            'best_class_acc': None
        }


    for epoch in range(1, num_epochs + 1):
        if pseudo:
            unlabelIter = iter(dataloaders['unlabel'])
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        if epoch > 20:  # increase the weight of pseudo label when model becomes valid
            pseudo_para = 0.2 + pseudo_para
        elif epoch > 40:
            pseudo_para = 0.5

        for phase in ['train', 'val']:
            epoch_loss = 0
            epoch_acc = 0
            class_loss = [0] * len(class_names)
            class_acc = [0] * len(class_names)
            class_cnt = [0] * len(class_names)

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    losses = loss_fn(outputs, labels)
                    loss = losses.mean()

                    try:
                        loss += pseudo_para * un_loss
                    except NameError:
                        pass

                    if pseudo:  # old model
                        un_inputs, un_labels = next(unlabelIter)
                        un_inputs = un_inputs.to(device)
                        un_outputs_curr = model(un_inputs)
                        _, un_labels = torch.max(un_outputs_curr, 1)
                        un_labels = un_labels.to(device)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if batch % batch_per_disp == 0:
                            batch_loss = loss.item()
                            batch_acc = torch.sum(preds == labels.data).item() / inputs.size(0)
                            try:
                                print('batch %d: loss %.3f (unloss %.3f) | acc %.3f' % (batch, batch_loss,un_loss, batch_acc))
                            except:
                                print('batch %d: loss %.3f | acc %.3f' % (batch, batch_loss, batch_acc))

                    if pseudo:  # new model
                        un_outputs = model(un_inputs)
                        _, un_preds = torch.max(un_outputs, 1)
                        un_losses = loss_fn(un_outputs, un_labels)
                        un_loss = un_losses.mean()
                        dataset_sizes['unlabel'] += len(un_outputs)

                if phase == 'val':
                    helper_class_loss_acc(class_loss, class_acc, class_cnt, losses, preds, labels)

                epoch_loss += loss.item() * inputs.size(0)
                epoch_acc += torch.sum(preds == labels.data).item()

            if phase == 'train':
                scheduler.step()

           # if pseudo:
                #dataset_sizes['train'] += dataset_sizes['unlabel']

            epoch_loss = epoch_loss / dataset_sizes[phase]
            epoch_acc = epoch_acc / dataset_sizes[phase]
            print('Epoch %d: %s loss %.3f | %s acc %.3f'
                  % (epoch, phase, epoch_loss, phase, epoch_acc))

            # generate pseudo label for the next training
            if pseudo:
                un_inputs, un_labels = next(unlabelIter)
                un_inputs = un_inputs.to(device)
                un_outputs_curr = model(un_inputs)
                _, un_labels = torch.max(un_outputs_curr, 1)
                un_labels = un_labels.to(device)

        if phase == 'val' and epoch_loss < result['best_loss']:
                result['best_loss'] = epoch_loss
                result['best_acc'] = epoch_acc
                class_cnt = np.array(class_cnt)
                result['val_class_cnt'] = class_cnt
                result['best_class_loss'] = np.array(class_loss, dtype=np.float) / class_cnt
                result['best_class_acc'] = np.array(class_acc, dtype=np.float) / class_cnt
                result['best_model'] = copy.deepcopy(model.state_dict())
    return result


def train_model_crossval(data_transforms, kfold_dir,  train_cfg, model_cfg, optimizer_cfg, scheduler_cfg,
                         loss_fn=nn.CrossEntropyLoss(reduction='none'), cv=True, pseudo=False, pseudo_para=0, test_dir = None):
    kfold_datasets = []

    for k in safe_listdir(kfold_dir):
        kfold_datasets.append(datasets.ImageFolder(os.path.join(kfold_dir, k)))

    kfold_result = []
    K = len(kfold_datasets)

    for i in range(K):
        print('K_Fold CV {}/{}'.format(i + 1, K))
        print('=' * 10)
        train_sets = kfold_datasets[:i] + kfold_datasets[i + 1:]
        for s in train_sets:
            s.transform = data_transforms['train']

        val_set = kfold_datasets[i]
        val_set.transform = data_transforms['val']

        if pseudo:
            test_set = datasets.ImageFolder(os.path.join(test_dir))
            test_set.transform = data_transforms['train']  # same transformation because of training purpose
            image_datasets = {'train': ConcatDataset(train_sets),
                              'val': val_set,
                              'unlabel': test_set}

        else:
            image_datasets = {
                'train': ConcatDataset(train_sets),
                'val': val_set}

        class_names, dataloaders, dataset_sizes = helper_dataloaders(image_datasets, train_cfg['batch_size'])
        model, optimizer, scheduler = helper_train(model_cfg, optimizer_cfg, scheduler_cfg)

        result = train_model(class_names, dataset_sizes, dataloaders, model, loss_fn, optimizer, scheduler,
                             model_cfg['device'], train_cfg['num_epochs'], train_cfg['batch_per_disp'], pseudo,
                             pseudo_para)

        kfold_result.append(result)

        ckpoint = {
            'kfold_result': kfold_result,
            'class_names': class_names
        }
        torch.save(ckpoint, 'ckpoint.pt')

        if not cv:
            break

    return ckpoint


if __name__ == '__main__':
    pass
