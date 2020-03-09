from torchvision import transforms
import preprocess
import torch

data_transforms = dict()
data_transforms['train'] = transforms.Compose([
    preprocess.PadToSquare(),
    transforms.RandomAffine(degrees=180, translate=(0.2, 0.2),
                            scale=(0.7, 1.5), shear=30, fillcolor=(255, 255, 255)),
    transforms.ColorJitter(contrast=[0.5, 1.5]),
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


train_cfg = {
    'num_epochs': 50,
    'batch_size': 64,
    'batch_per_disp': 100
}

model_cfg = {
    'backbone': 'resnet18',
    'pretrained': True,
    'fc_hidden_dim': [256, 256],
    'tune_conv': True,
    'num_classes': 121,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

optimizer_cfg = {
    'lr': 5e-4,
}

scheduler_cfg = {
    'step_size': 10,
    'gamma': 0.5
}



