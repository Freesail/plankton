import torch
from model import helper_model
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from model import safe_listdir
import os
import pandas as pd
import tqdm
import numpy as np


class TestDataset(VisionDataset):
    def __init__(self, root, loader=default_loader, transform=None,
                 target_transform=None):
        super(TestDataset, self).__init__(root, transform=transform,
                                          target_transform=target_transform)
        self.loader = loader
        self.samples = safe_listdir(self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name = self.samples[index]
        path = os.path.join(self.root, name)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, name


def test_model(data_transfoms, test_dir, ckpoint_path, model_cfg, class_names, submission):
    test_dataset = TestDataset(test_dir, transform=data_transfoms['test'])
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=16)
    out_df = pd.DataFrame(0, index=test_dataset.samples, columns=class_names)
    model, _, _ = helper_model(**model_cfg)
    ckpoint = torch.load(ckpoint_path, map_location=model_cfg['device'])
    # todo
    model.load_state_dict(ckpoint['kfold_result'][0]['best_model'])
    model.eval()
    for inputs, names in tqdm.tqdm(test_dataloader):
        inputs = inputs.to(model_cfg['device'])
        with torch.no_grad():
            outputs = model(inputs)
            prob = torch.nn.functional.softmax(outputs, dim=1)
        for i, name in enumerate(names):
            out_df.loc[name, :] = prob[i, :].cpu().numpy()

    out_df.to_csv(submission, index_label='image')


if __name__ == '__main__':
    from config import data_transforms, model_cfg

    test_dir = './data/raw_data/mini_test'
    ckpoint_path = './output/ckpoint.pt'
    class_names = datasets.ImageFolder('./data/raw_data/train').classes
    test_model(data_transforms, test_dir, ckpoint_path, model_cfg, class_names,
               submission='./output/submission.csv')
