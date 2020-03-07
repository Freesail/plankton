from torchvision.models import *
from torch import nn
import torch
import copy
from data import safe_listdir
from torchvision import transforms, datasets
import os
from torch.utils.data.dataset import ConcatDataset
from torch.optim.lr_scheduler import StepLR


def helper_mlp(in_dim, hiddent_dim, out_dim):
    layers = []
    hiddent_dim.insert(0, in_dim)
    hiddent_dim.append(out_dim)

    for i in range(len(hiddent_dim) - 1):
        layers.append(nn.Linear(hiddent_dim[i], hiddent_dim[i + 1]))
        layers.append(nn.ReLU())
    layers.pop()
    return nn.Sequential(*tuple(layers))


def helper_model(backbone, fc_hidden_dim, num_classes, device):
    model_conv = eval('%s(pretrained=True)' % backbone)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = helper_mlp(num_ftrs, fc_hidden_dim, num_classes)
    model_conv = model_conv.to(device)
    return model_conv


def helper_train(model_cfg, optimizer_cfg, scheduler_cfg):
    model = helper_model(**model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_cfg)
    scheduler = StepLR(optimizer, **scheduler_cfg)
    return model, optimizer, scheduler


def helper_dataloaders(image_datasets, batch_size):
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes


def train_model(dataset_sizes,
                dataloaders,
                model, loss_fn, optimizer, scheduler, device,
                num_epochs=50, batch_per_disp=128):
    best_val_model = copy.deepcopy(model.state_dict())
    best_val_loss = 100
    best_val_acc = 0
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            epoch_loss = 0
            epoch_acc = 0

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
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if batch % batch_per_disp == 0:
                            batch_loss = loss.item()
                            batch_acc = torch.sum(preds == labels.data).item() / inputs.size(0)
                            print('batch %d: loss %.3f | acc %.3f' % (batch, batch_loss, batch_acc))

                epoch_loss += loss.item() * inputs.size(0)
                epoch_acc += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = epoch_loss / dataset_sizes[phase]
            epoch_acc = epoch_acc / dataset_sizes[phase]
            print('Epoch %d: %s loss %.3f | %s acc %.3f'
                  % (epoch, phase, epoch_loss, phase, epoch_acc))

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_val_acc = epoch_acc
                best_val_model = copy.deepcopy(model.state_dict())

    return best_val_loss, best_val_acc, best_val_model


def train_model_val(data_transforms, data_dir, train_cfg,
                    model_cfg, optimizer_cfg, scheduler_cfg,
                    loss_fn=nn.CrossEntropyLoss()):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders, dataset_sizes = helper_dataloaders(image_datasets, train_cfg['batch_size'])
    model, optimizer, scheduler = \
        helper_model(model_cfg, optimizer_cfg, scheduler_cfg)

    print('Training starts ...')
    best_val_loss, best_val_acc, best_val_model = \
        train_model(dataset_sizes, dataloaders,
                    model, loss_fn, optimizer, scheduler,
                    model_cfg['device'],
                    train_cfg['num_epochs'],
                    train_cfg['batch_per_disp'])
    print('Training ends')
    print('Best val loss %.3f | acc: %.3f' % (best_val_loss, best_val_acc))
    ckpoint = {
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_val_model': best_val_model
    }
    torch.save(ckpoint, 'ckpoint.pt')
    return best_val_loss, best_val_acc, best_val_model


def train_model_crossval(data_transforms, kfold_dir, train_cfg,
                         model_cfg, optimizer_cfg, scheduler_cfg,
                         loss_fn=nn.CrossEntropyLoss()):
    kfold_datasets = []
    for k in safe_listdir(kfold_dir):
        kfold_datasets.append(datasets.ImageFolder(os.path.join(kfold_dir, k)))

    kfold_val_loss = []
    kfold_val_acc = []
    kfold_val_model = []
    
    K = len(kfold_datasets)
    for i in range(K):
        print('K_Fold CV {}/{}'.format(i+1, K))
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
        dataloaders, dataset_sizes = helper_dataloaders(image_datasets, train_cfg['batch_size'])
        model, optimizer, scheduler = \
            helper_model(model_cfg, optimizer_cfg, scheduler_cfg)

        best_val_loss, best_val_acc, best_val_model = \
            train_model(dataset_sizes, dataloaders,
                        model, loss_fn, optimizer, scheduler,
                        model_cfg['device'],
                        train_cfg['num_epochs'],
                        train_cfg['batch_per_disp'])

        kfold_val_loss.append(best_val_loss)
        kfold_val_acc.append(best_val_acc)
        kfold_val_model.append(best_val_model)

        ckpoint = {
            'kfold_val_loss': kfold_val_loss,
            'kfold_val_acc': kfold_val_acc,
            'kfold_val_model': kfold_val_model
        }
        torch.save(ckpoint, 'ckpoint.pt')
    return kfold_val_loss, kfold_val_acc, kfold_val_model


if __name__ == '__main__':
    data_transforms = dict()

    data_transforms['train'] = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    data_transforms['val'] = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor()
    ])
    device = torch.device('cpu')
    # train_model_cv(data_transforms, kfold_dir='./data/raw_data/kfold', batch_size=2, device=device)
