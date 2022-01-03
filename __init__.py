import sys
from platform import system

import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

sys.path.append('src')  # Do not remove this code!

RUN_DIR = 'runs'  # directory name for storing *.pt & *.csv. & storing log info on tensorboard.
# (DO NOT INSERT '/' AT THE END OF LINE)

NEWLINE = '\n' if system == 'Windows' else ''  # Recommendation: (for win) '\n' (for linux) ''


def get_manual_transform_list(mode, transform_list_name, mean, std):
    if mode == 'train':
        transform_list_dir = {
            'CIFAR': [
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ],  # you can add your own manual transform list!
        }
    else:  # for 'valid' and 'test'
        transform_list_dir = {
            'CIFAR': [
                transforms.ToTensor(),
            ],  # you can add your own manual transform list!
        }

    transform_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if transform_list_dir.get(transform_list_name):
        transform_list = transform_list_dir[transform_list_name] + [transforms.Normalize(mean, std)]
    return transform_list


def get_optimizer(model, lr):
    return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)


def get_lr_scheduler(optimizer, lr_step, lr_step_gamma):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step, gamma=lr_step_gamma)


def get_loss_function(loss_function_name, device):
    if loss_function_name is None:
        return None
    if loss_function_name == 'CE':  # default
        return nn.CrossEntropyLoss()
        
    elif loss_function_name == 'CCE':  # complement cross entropy (PRL)
        from src.my_criterion.cce import CCE
        return CCE(device, balancing_factor=1)
    # [Note]    if you want to use your own loss function,
    #           add code as follows:
    #           e.g.    elif loss_function__name == 'your_own_function_name':
    #                       return your_own_function()
    else:
        print(f'[Warning] Invalid loss function name is given: "{loss_function_name}". '
              'nn.CrossEntropyLoss() is returned instead. '
              'Try to check whether the loss function name is correct.')
        return nn.CrossEntropyLoss()
