from torchvision.models import *
from torch import nn
import torch
import copy
from data import safe_listdir
from torchvision import transforms, datasets
import os
from torch.utils.data.dataset import ConcatDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np


def helper_mlp(in_dim, hiddent_dim, out_dim):
    layers = []
    hiddent_dim.insert(0, in_dim)
    hiddent_dim.append(out_dim)

    for i in range(len(hiddent_dim) - 1):
        layers.append(nn.Linear(hiddent_dim[i], hiddent_dim[i + 1]))
        layers.append(nn.ReLU())
    layers.pop()
    return nn.Sequential(*tuple(layers))


def helper_model(backbone, pretrained, fc_hidden_dim, tune_conv, num_classes, device):
    if backbone == 'resnet18':
        model_conv = resnet18(pretrained=pretrained)
        if pretrained and (not tune_conv):
            for param in model_conv.parameters():
                param.requires_grad = False
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = helper_mlp(num_ftrs, fc_hidden_dim, num_classes)
        fc_params = model_conv.fc.parameters()
        model_conv = model_conv.to(device)

        ignored_params = list(map(id, model_conv.fc.parameters()))
        conv_params = filter(lambda p: id(p) not in ignored_params,
                             model_conv.parameters())
    else:
        raise NotImplementedError

    return model_conv, conv_params, fc_params


def helper_train(model_cfg, optimizer_cfg, scheduler_cfg):
    model, conv_params, fc_params = helper_model(**model_cfg)
    if model_cfg['pretrained'] and (not model_cfg['tune_conv']):
        optimizer = torch.optim.Adam(model.fc.parameters(), **optimizer_cfg)
    else:
        optimizer = torch.optim.Adam(
            [{'params': conv_params,
              'lr': optimizer_cfg['conv_lr_ratio'] * optimizer_cfg['lr']},
             {'params': fc_params}],
            **optimizer_cfg)
    scheduler = StepLR(optimizer, **scheduler_cfg)
    return model, optimizer, scheduler


def helper_dataloaders(image_datasets, batch_size):
    class_names = image_datasets['val'].classes
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=16)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
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
                num_epochs=50, batch_per_disp=128):
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
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

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

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if batch % batch_per_disp == 0:
                            batch_loss = loss.item()
                            batch_acc = torch.sum(preds == labels.data).item() / inputs.size(0)
                            print('batch %d: loss %.3f | acc %.3f' % (batch, batch_loss, batch_acc))

                if phase == 'val':
                    helper_class_loss_acc(class_loss, class_acc, class_cnt, losses, preds, labels)

                epoch_loss += loss.item() * inputs.size(0)
                epoch_acc += torch.sum(preds == labels.data).item()

            if phase == 'train':
                scheduler.step()

            epoch_loss = epoch_loss / dataset_sizes[phase]
            epoch_acc = epoch_acc / dataset_sizes[phase]
            print('Epoch %d: %s loss %.3f | %s acc %.3f'
                  % (epoch, phase, epoch_loss, phase, epoch_acc))

            if phase == 'val' and epoch_loss < result['best_loss']:
                result['best_loss'] = epoch_loss
                result['best_acc'] = epoch_acc
                class_cnt = np.array(class_cnt)
                result['val_class_cnt'] = class_cnt
                result['best_class_loss'] = np.array(class_loss, dtype=np.float) / class_cnt
                result['best_class_acc'] = np.array(class_acc, dtype=np.float) / class_cnt
                result['best_model'] = copy.deepcopy(model.state_dict())
    return result


# def train_model_val(data_transforms, data_dir, train_cfg,
#                     model_cfg, optimizer_cfg, scheduler_cfg,
#                     loss_fn=nn.CrossEntropyLoss(reduction='none')):
#     image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                       for x in ['train', 'val']}
#     class_names, dataloaders, dataset_sizes = helper_dataloaders(image_datasets, train_cfg['batch_size'])
#     model, optimizer, scheduler = \
#         helper_train(model_cfg, optimizer_cfg, scheduler_cfg)
#
#     print('Training starts ...')
#     best_val_loss, best_val_acc, best_val_class_loss, best_val_class_acc, best_val_model = \
#         train_model(class_names, dataset_sizes, dataloaders,
#                     model, loss_fn, optimizer, scheduler,
#                     model_cfg['device'],
#                     train_cfg['num_epochs'],
#                     train_cfg['batch_per_disp'])
#     print('Training ends')
#     print('Best val loss %.3f | acc: %.3f' % (best_val_loss, best_val_acc))
#     ckpoint = {
#         'best_val_loss': best_val_loss,
#         'best_val_acc': best_val_acc,
#         'best_val_model': best_val_model,
#         'config': (data_transforms, train_cfg, model_cfg, optimizer_cfg, scheduler_cfg)
#     }
#     torch.save(ckpoint, 'ckpoint.pt')
#     return best_val_loss, best_val_acc


def train_model_crossval(data_transforms, kfold_dir, train_cfg,
                         model_cfg, optimizer_cfg, scheduler_cfg,
                         loss_fn=nn.CrossEntropyLoss(reduction='none'), cv=True):
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
        image_datasets = {
            'train': ConcatDataset(train_sets),
            'val': val_set
        }
        class_names, dataloaders, dataset_sizes = helper_dataloaders(image_datasets, train_cfg['batch_size'])
        model, optimizer, scheduler = \
            helper_train(model_cfg, optimizer_cfg, scheduler_cfg)

        result = \
            train_model(class_names, dataset_sizes, dataloaders,
                        model, loss_fn, optimizer, scheduler,
                        model_cfg['device'],
                        train_cfg['num_epochs'],
                        train_cfg['batch_per_disp'])

        kfold_result.append(result)

        ckpoint = {
            'kfold_result': kfold_result,
            'config': (data_transforms, train_cfg, model_cfg, optimizer_cfg, scheduler_cfg)
        }
        torch.save(ckpoint, 'ckpoint.pt')

        if not cv:
            break

    return kfold_result


if __name__ == '__main__':
    pass
