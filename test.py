import datetime
import os.path
from warnings import filterwarnings

import torch

from src.my_utils import util
from __init__ import *
from src import loader, model, iterator, parser

''' [Note]
This code logic is similar to [train.py].
However, there are some changes in [train.py].
Please see '⭐⭐⭐' comments for the differences! (I.e. '# ⭐⭐⭐')
'''

if __name__ == '__main__':
    # Ignore Warning Messages
    filterwarnings('ignore')

    # Parser
    my_parser = parser.Parser(mode='test')  # ⭐
    my_args = my_parser.parser.parse_args()

    # Tag
    cur_time = datetime.datetime.now().strftime('%y%m%d|%H:%M:%S')
    tag_name = f'TEST-{os.path.basename(my_args.checkpoint)[6:-3]}-{cur_time}'  # ⭐
    print(f'{tag_name} experiment has been started.')

    # Loader (Train / Valid)
    my_loader = loader.Loader(my_args.dataset_dir, my_args.batch_size,
                              my_args.mean, my_args.std,
                              my_args.auto_mean_std,  # if my_args.auto_mean_std is True,
                                                      # my_args.mean, my_args.std become ignored.
                              my_args.transform_list_name)
    my_test_loader = my_loader.get_loader(mode='test', shuffle=False)  # ⭐

    # Initialization
    my_model = model.model(my_args.network_name, my_loader.num_classes, pretrained=False)
    my_model.load_state_dict(torch.load(my_args.checkpoint))  # ⭐⭐⭐
    my_device = 'cpu' if my_args.gpu_index == -1 else f'cuda:{my_args.gpu_index}'

    # Iterator
    my_iterator = iterator.Iterator(my_model, None, None, None, my_loader.num_classes, tag_name,
                                    my_device, None, None, None,
                                    my_args.store_confusion_matrix, my_args.store_logits)  # ⭐
    my_iterator.set_loader('test', my_test_loader)  # ⭐

    # Test
    my_iterator.test()  # ⭐⭐⭐

    util.store_setup_txt(f'{RUN_DIR}/{tag_name}/setup-test.txt', my_args)

    print(f'{tag_name} experiment has been done.')
