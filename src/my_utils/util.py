import itertools
import random

from matplotlib import pyplot as plt
from torch import manual_seed, cuda, backends
import numpy as np
from sklearn.metrics import confusion_matrix


class Meter:
    def __init__(self):
        self.values, self.avg, self.sum, self.cnt = [], 0, 0, 0

    def reset(self):
        self.values, self.avg, self.sum, self.cnt = [], 0, 0, 0

    def update(self, value, k=1):
        self.values.append(value)
        self.sum += value
        self.cnt += k
        self.avg = self.sum / self.cnt


def fix_random_seed(seed=1234):
    # Ref.: https://github.com/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    backends.cudnn.deterministic = True


def store_setup_txt(path, my_args):
    arg_keys = list(filter(lambda x: x[0] != '_', dir(my_args)))
    if my_args.auto_mean_std:
        arg_keys.remove('mean')
        arg_keys.remove('std')
    with open(path, 'w') as f:
        for arg_key in arg_keys:
            f.write(f'{arg_key}: {my_args.__getattribute__(f"{arg_key}")} \n')
        f.flush()


def create_confusion_matrix(y_trues, y_preds, num_of_classes,
                            class_names=None, threshold=15, figsize=(8, 6), cmap=plt.cm.Blues, title=''):
    cf_matrix = confusion_matrix(y_trues, y_preds)
    normalized_cf_matrix = cf_matrix / np.sum(cf_matrix) * num_of_classes
    # normalized_cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(normalized_cf_matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    bool_postfix_index = True
    if num_of_classes <= threshold:
        bool_postfix_index = False
        labels = class_names if class_names else np.arange(num_of_classes)
        plt.xticks(np.arange(num_of_classes), labels, rotation=45)
        plt.yticks(np.arange(num_of_classes), labels)

        # plotting probabilities: p(prediction=i|ground_truth=j) for all classes i and j.
        txt_color_threshold = normalized_cf_matrix.max() / 2.
        for i, j in itertools.product(np.arange(normalized_cf_matrix.shape[0]),
                                      np.arange(normalized_cf_matrix.shape[1])):
            plt.text(j, i, f'{normalized_cf_matrix[i, j]: .2f}',
                     horizontalalignment='center',
                     color='white' if normalized_cf_matrix[i, j] > txt_color_threshold else 'black')

    plt.tight_layout()
    plt.xlabel('Predicted Class' + ('' if not bool_postfix_index else ' Index'))
    plt.ylabel('Ground Truth Class' + ('' if not bool_postfix_index else ' Index'))

    return fig
