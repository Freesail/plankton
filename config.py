from torchvision import transforms
import preprocess
import torch

data_transforms = dict()
data_transforms['train'] = transforms.Compose([
    preprocess.PadToSquare(),
    transforms.RandomRotation(degrees=180),
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
    'batch_size': 128,
    'batch_per_disp': 50
}

model_cfg = {
    'backbone': 'resnet18',
    'pretrained': False,
    'fc_hidden_dim': [512, 512],
    'tune_conv': False,
    'num_classes': 121,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

optimizer_cfg = {
    'lr': 3e-4,
}

scheduler_cfg = {
    'step_size': 10,
    'gamma': 0.5
}



