import ast
import os
import pickle

import torch
from torch.utils import data
from torchvision import datasets

from __init__ import *
from dataset import CustomDataset


class Loader:
    mean_pkl, std_pkl = 'mean.pkl', 'std.pkl'

    def __init__(self, dataset_path, batch_size=1,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 auto_mean_std=False,
                 transform_list_name='',
                 num_workers=2):
        self.dataset_path = dataset_path
        self.dataset_dir = self.__get_dataset_dir()
        self.classes = self.__get_classes()
        self.num_classes = len(self.classes)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = ast.literal_eval(mean) if type(mean) == str else mean
        self.std = ast.literal_eval(std) if type(std) == str else std
        if auto_mean_std:
            self.mean, self.std = self.__get_train_mean_std()
        # see '__init__.py' for transform_list!
        train_transform_list = get_manual_transform_list('train', transform_list_name, self.mean, self.std)
        eval_transform_list = get_manual_transform_list('eval', transform_list_name, self.mean, self.std)
        self.transform_dir = {'train': train_transform_list,
                              'valid': eval_transform_list,
                              'test': eval_transform_list}

    def __get_dataset_dir(self):
        modes = ['train', 'valid', 'test']
        dataset_dir = {data_directory: os.path.join(self.dataset_path, data_directory)
                       for data_directory in modes
                       if os.path.isdir(os.path.join(self.dataset_path, data_directory))}
        # 'train' and 'validation' directory must be exist!
        assert dataset_dir['train'] and dataset_dir['valid']
        return dataset_dir

    def __get_classes(self):
        return [class_dir for class_dir in os.listdir(self.dataset_dir['train'])
                if not os.path.isfile(class_dir)]

    def __get_train_mean_std(self):
        Loader.mean_pkl, Loader.std_pkl = os.path.join(self.dataset_path, Loader.mean_pkl), \
                                          os.path.join(self.dataset_path, Loader.std_pkl),
        if os.path.exists(Loader.mean_pkl) and os.path.exists(Loader.std_pkl):
            with open(Loader.mean_pkl, 'rb') as pkl:
                mean = pickle.load(pkl)
            with open(Loader.std_pkl, 'rb') as pkl:
                std = pickle.load(pkl)
        else:
            print('Hold a sec for calculating mean/std of training examples.')
            train_set = datasets.ImageFolder(root=self.dataset_dir['train'],
                                             transform=transforms.ToTensor())
            loader = data.DataLoader(train_set, batch_size=self.batch_size,
                                     shuffle=False, num_workers=0)
            mean, std = torch.zeros(3), torch.zeros(3)
            for inputs, targets in loader:
                for i in range(3):  # R, G, B
                    mean[i] += inputs[:, i, :, :].mean()
                    std[i] += inputs[:, i, :, :].std()
            mean /= len(loader)
            std /= len(loader)
            # memo for future use
            with open(Loader.mean_pkl, 'wb') as pkl:
                pickle.dump(mean, pkl)
            with open(Loader.std_pkl, 'wb') as pkl:
                pickle.dump(std, pkl)
        return mean, std

    def get_loader(self, mode, shuffle=True):
        assert self.dataset_dir[mode]  # mode: ['train' | 'valid' | 'test']
        # dataset = datasets.ImageFolder(root=self.dataset_dir[mode],
        #                                transform=transforms.Compose(self.transform_dir[mode]))
        dataset = CustomDataset(dataset_path=self.dataset_dir[mode],
                                transform=transforms.Compose(self.transform_dir[mode]))
        return data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
