import os
import shutil
import tqdm
from random import shuffle
import numpy as np


def safe_listdir(path):
    f = os.listdir(path)
    try:
        f.remove('.DS_Store')
    except ValueError:
        pass
    return f


def ignore_file(d, files):
    return [f for f in files if os.path.isfile(os.path.join(d, f))]


class PlanktonData:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.raw_train_dir = os.path.join(raw_data_dir, 'train')
        self.labels = safe_listdir(self.raw_train_dir)
        self.n_labels = len(self.labels)
        self.n_train_samples = len(self.get_train_samples())

    def get_train_samples(self):
        samples = []
        for c in self.labels:
            c_dir = os.path.join(self.raw_train_dir, c)
            samples.extend(safe_listdir(c_dir))
        return samples

    def get_train_samples_with_label(self):
        samples = {}
        for c in self.labels:
            c_dir = os.path.join(self.raw_train_dir, c)
            samples[c] = safe_listdir(c_dir)
        return samples

    def split_kfold(self, k_fold, dst_dir):
        # create directory
        for k in range(k_fold):
            dst_dir_k = os.path.join(dst_dir, 'fold_%d' % k)
            try:
                shutil.copytree(self.raw_train_dir, dst_dir_k, ignore=ignore_file)
            except FileExistsError:
                pass

        # copy data
        k_fold_n = np.zeros(k_fold, dtype=np.int32)
        samples_with_label = self.get_train_samples_with_label()
        print('splitting k_fold ...')
        for c in self.labels:
            samples = samples_with_label[c]
            nc = len(samples)
            if nc % k_fold != 0:
                n_per_fold = int(nc / k_fold)
            else:
                n_per_fold = nc / k_fold
            shuffle(samples)
            folds = list(range(k_fold))
            shuffle(folds)
            for i, k in enumerate(folds):
                if i == k_fold - 1:
                    samples_this_fold = samples[int(i * n_per_fold):]
                else:
                    samples_this_fold = samples[int(i * n_per_fold):int((i + 1) * n_per_fold)]
                for s in samples_this_fold:
                    src = os.path.join(self.raw_train_dir, c, s)
                    dst = os.path.join(dst_dir, 'fold_%d' % k, c)
                    shutil.copy2(src, dst)
                k_fold_n[k] = k_fold_n[k] + len(samples_this_fold)

        print('samples in k_fold: ', k_fold_n)
        print('total samples in kfold: ', k_fold_n.sum())
        print('total samples in raw train set: ', self.n_train_samples)


if __name__ == '__main__':
    plankton = PlanktonData(raw_data_dir='./data/raw_data')
    plankton.split_kfold(k_fold=5, dst_dir='./data/raw_data/kfold')

