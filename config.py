from torchvision import transforms
import preprocess
import torch
import numpy as np

data_transforms = dict()
data_transforms['train'] = transforms.Compose([
    preprocess.PadToSquare(),
    transforms.RandomAffine(degrees=180, translate=(0.1, 0.1),
                            scale=(0.7, 1.2), shear=20, fillcolor=(255, 255, 255)),
    transforms.ColorJitter(contrast=[0.7, 1.1]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
data_transforms['val'] = transforms.Compose([
    preprocess.PadToSquare(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
data_transforms['test'] = transforms.Compose([
    preprocess.PadToSquare(),
    transforms.RandomAffine(degrees=180, translate=(0.1, 0.1),
                            scale=(0.7, 1.2), shear=15, fillcolor=(255, 255, 255)),
    transforms.ColorJitter(contrast=[0.7, 1.1]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# target_transform = preprocess.NoisyLabel(np.load('./plankton/T_matrix.npy'))

train_cfg = {
    'num_epochs': 30,
    'batch_size': 128,
    'batch_per_disp': 100
}

model_cfg = {
    'backbone': 'resnet18',
    'pretrained': True,
    'fc_hidden_dim': [512, 512],
    'tune_conv': True,
    'fc_dropout': 0.5,
    'is_bayes': False,
    'num_classes': 121,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

optimizer_cfg = {
    'lr': 5e-4,
    'conv_lr_ratio': 0.4
}

scheduler_cfg = {
    'step_size': 15,
    'gamma': 0.3
}

# def pseudo_scheduler(epoch):
#     if epoch < 3:
#         return 0.0
#
#     if 3 <= epoch <= 25:
#         return (epoch - 5) * 0.3 / 22
#
#     if epoch > 25:
#         return 0.3
