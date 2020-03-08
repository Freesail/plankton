from torchvision import transforms
import preprocess
import torch

data_transforms = dict()
data_transforms['train'] = transforms.Compose([
    preprocess.PadToSquare(),
    transforms.Resize((224, 224)),
    transforms.RandomAffine(180, translate=(0.1, 0.1), scale=(1/1.3, 1.3), shear=20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
data_transforms['val'] = transforms.Compose([
    preprocess.PadToSquare(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


train_cfg = {
    'num_epochs': 30,
    'batch_size': 128,
    'batch_per_disp': 50
}

model_cfg = {
    'backbone': 'resnet18',
    'fc_hidden_dim': [256, 256],
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



