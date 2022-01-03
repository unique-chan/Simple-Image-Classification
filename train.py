import datetime
import os.path
from warnings import filterwarnings

from src.my_utils import util
from __init__ import *
from src import loader, model, iterator, parser

if __name__ == '__main__':
    # Ignore Warning Messages
    filterwarnings('ignore')

    # Parser
    my_parser = parser.Parser(mode='train')
    my_args = my_parser.parser.parse_args()

    # Tag
    cur_time = datetime.datetime.now().strftime('%y%m%d|%H:%M:%S')
    tag_name = f'TRAIN-[{my_args.tag}|{os.path.basename(my_args.dataset_dir)}|{my_args.network_name}]|{cur_time}'
    print(f'{tag_name} experiment has been started.')

    # Loader (Train / Valid)
    my_loader = loader.Loader(my_args.dataset_dir, my_args.batch_size,
                              my_args.mean, my_args.std,
                              my_args.auto_mean_std,  # if my_args.auto_mean_std is True,
                                                      # my_args.mean, my_args.std become ignored.
                              my_args.transform_list_name)
    my_train_loader = my_loader.get_loader(mode='train', shuffle=True)
    my_valid_loader = my_loader.get_loader(mode='valid', shuffle=False)

    # Initialization
    my_model = model.model(my_args.network_name, my_loader.num_classes, pretrained=False)
    my_device = 'cpu' if my_args.gpu_index == -1 else f'cuda:{my_args.gpu_index}'
    # see '__init__.py' for my_optimizer & get_lr_scheduler
    my_optimizer = get_optimizer(my_model, my_args.lr)
    my_lr_scheduler = get_lr_scheduler(my_optimizer, my_args.lr_step, my_args.lr_step_gamma) \
        if my_args.lr_step else None

    # Iterator
    my_iterator = iterator.Iterator(my_model, my_optimizer, my_lr_scheduler, my_args.lr_warmup_epochs,
                                    my_loader.num_classes, tag_name,
                                    my_device, my_args.loss_function,
                                    my_args.store_weights, my_args.store_loss_acc_log,
                                    my_args.store_confusion_matrix, my_args.store_logits)
    my_iterator.set_loader('train', my_train_loader)
    my_iterator.set_loader('valid', my_valid_loader)

    # Training and Validation
    for cur_epoch in range(0, my_args.epochs):
        my_iterator.train(cur_epoch=cur_epoch)
        my_iterator.valid(cur_epoch=cur_epoch)

    if my_args.store_weights:
        my_iterator.store_model()

    util.store_setup_txt(f'{RUN_DIR}/{tag_name}/setup-train-val.txt', my_args)

    print(f'{tag_name} experiment has been done.')
