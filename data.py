import os
import shutil
import tqdm
from random import shuffle
import numpy as np
import random


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

    def split_kfold(self, k_fold, dst_dir, oversample=False):
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
        for c in tqdm.tqdm(self.labels):
            samples = samples_with_label[c]
            nc = len(samples)
            nf = int(nc / k_fold)
            n_per_fold = np.ones(k_fold, np.int) * nf
            n_per_fold[:int(nc - nf * k_fold)] = n_per_fold[:int(nc - nf * k_fold)] + 1
            n_per_fold = np.cumsum(n_per_fold)
            n_per_fold = np.insert(n_per_fold, 0, 0)
            shuffle(samples)
            folds = list(range(k_fold))
            shuffle(folds)
            for i, k in enumerate(folds):
                samples_this_fold = samples[n_per_fold[i]:n_per_fold[i + 1]]
                for s in samples_this_fold:
                    src = os.path.join(self.raw_train_dir, c, s)
                    dst = os.path.join(dst_dir, 'fold_%d' % k, c)
                    shutil.copy2(src, dst)
                k_fold_n[k] = k_fold_n[k] + len(samples_this_fold)

        if oversample:
            for k in tqdm.tqdm(range(k_fold)):
                for c in self.labels:
                    dst = os.path.join(dst_dir, 'fold_%d' % k, c)
                    self.oversample_dir(dst)

    @staticmethod
    def oversample_dir(dst_dir, threshold=100):
        samples = safe_listdir(dst_dir)
        n_samples = len(samples)
        if n_samples < threshold:
            for i in range(int(threshold / n_samples) - 1):
                for s in samples:
                    shutil.copyfile(os.path.join(dst_dir, s), os.path.join(dst_dir, 'copy_%d_%s' % (i, s)))

            random_samples = random.sample(samples, threshold % n_samples)
            for s in random_samples:
                shutil.copyfile(os.path.join(dst_dir, s), os.path.join(dst_dir, 'COPY_%s' % s))

            assert len(safe_listdir(dst_dir)) == threshold


if __name__ == '__main__':
    plankton = PlanktonData(raw_data_dir='./data/raw_data')
    plankton.split_kfold(k_fold=7, dst_dir='./data/raw_data/kfold', oversample=False)
