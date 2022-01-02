import csv
import os

import torch
import torch.nn as nn

from src.my_utils import util, warmup_schduler
from __init__ import *

try:
    bool_tqdm = True
    import tqdm
except ImportError:
    bool_tqdm = False
    print('[Warning] Try to install tqdm for progress bar.')

try:
    bool_tb = True
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    bool_tb = False
    print('[Warning] Try to install tensorboard for checking the status of learning.')

LOSS_ACC_STATE_FIELDS = ['epoch',
                         'train_loss', 'valid_loss',
                         'train_top1_acc', 'train_top5_acc', 'valid_top1_acc', 'valid_top5_acc']


class Iterator:
    def __init__(self, model, optimizer, lr_scheduler, lr_warmup_epochs, num_classes, tag_name,
                 device='cpu', loss_function_name='CE', store_weights=False, store_loss_acc_log=False,
                 store_confusion_matrix=False, store_logits=False):
        self.model = model
        self.optimizer = optimizer
        self.loader = {'train': None, 'valid': None, 'test': None}
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_warmup_scheduler = warmup_schduler.WarmUpLR(optimizer) if lr_warmup_epochs > 0 else None
        self.num_classes = num_classes
        self.device = device  # 'cpu', 'cuda:0', ...
        self.model.to(device)
        self.criterion = get_loss_function(loss_function_name, device)
        self.store_weights = store_weights
        self.store_loss_acc_log = store_loss_acc_log
        self.store_confusion_matrix = store_confusion_matrix
        self.store_logits = store_logits
        self.tag_name = tag_name
        self.best_valid_acc_state = {'top1_acc': 0, 'top5_acc': 0}
        if store_weights or store_loss_acc_log or store_logits:
            os.makedirs(f'{LOG_DIR}/{tag_name}', exist_ok=True)
        if store_weights:
            # [GOAL] store the best validation model during training.
            self.best_model_state_path = f'{LOG_DIR}/{tag_name}/{tag_name}_valid_best.pt'
            self.best_model_state_dict = self.model.state_dict()
        if store_loss_acc_log:
            # [GOAL] store train/valid loss & acc per each epoch during training.
            self.loss_acc_state = {field_name: 0 for field_name in LOSS_ACC_STATE_FIELDS}
            self.log_loss_acc_csv_path = f'{LOG_DIR}/{tag_name}/{tag_name}.csv'
            self.log_loss_acc_csv_writer = csv.DictWriter(open(self.log_loss_acc_csv_path, 'w', newline=NEWLINE),
                                                          fieldnames=LOSS_ACC_STATE_FIELDS)
            self.log_loss_acc_csv_writer.writeheader()
        if store_logits:
            # [GOAL] store output distributions per each epoch for all images in the current experiment.
            self.logits_root_path = f'{LOG_DIR}/{self.tag_name}/logits'
            self.logits_csv_writers = {}  # key: 'img_path' - value: csv_writer for corresponding key
        if store_confusion_matrix:
            # [Goal] store confusion matrix if so far best during validation
            self.confusion_matrix_root_path = f'{LOG_DIR}/{self.tag_name}/cf_matrix'
            os.makedirs(self.confusion_matrix_root_path, exist_ok=True)
        self.tb_writer = SummaryWriter(f'{RUN_DIR}/{self.tag_name}') if bool_tb else None

    def set_loader(self, mode, loader):
        self.loader[mode] = loader
        if mode == 'train' and self.lr_warmup_scheduler:
            self.lr_warmup_scheduler.set_total_iters(len(loader) * self.lr_warmup_epochs)

    def one_epoch(self, mode, cur_epoch):
        loader = self.loader[mode]
        meter = {'loss': util.Meter(), 'top1_acc': util.Meter(), 'top5_acc': util.Meter()}
        assert loader, f"No loader['{mode}'] exists. Pass the loader to the Iterator via set_loader()."
        tqdm_loader = tqdm.tqdm(loader, mininterval=0.1) if bool_tqdm else loader
        img_paths = []  # set of image(x) paths
        y_trues = []  # set of ground_truth
        y_preds = []  # set of prediction (classification result)
        y_dists = []  # set of predicted distribution
        for (img_path, x, y) in tqdm_loader:
            x, y = x.to(self.device), y.to(self.device)
            # predict
            y_dist = self.model(x)
            y_pred, [top1_acc, top5_acc] = \
                Iterator.__get_final_classification_results_and_topk_acc__(y_dist, y, top_k=(1, 5))
            # calculate loss
            if mode in ['train', 'valid']:
                loss = self.criterion(y_dist, y)
                if mode == 'train':
                    self.__optimize_model(cur_epoch, loss)
            # to store logits or confusion matrix, accumulate the prediction results!
            if self.store_logits or self.store_confusion_matrix:
                Iterator.__accumulate_predictions(img_path, img_paths, y, y_dist, y_dists, y_pred, y_preds, y_trues)
            # to print the log!
            if self.store_loss_acc_log:
                Iterator.__update_all_meters(loss if mode in ['train', 'valid'] else None,
                                             meter, top1_acc, top5_acc, k=y_dist.size(0))
            if bool_tqdm:
                self.__print_tqdm_log(cur_epoch, meter, mode, tqdm_loader)
        return meter['loss'].avg, meter['top1_acc'].avg * 100., meter['top5_acc'].avg * 100., \
               img_paths, y_trues, y_preds, y_dists

    def train(self, cur_epoch):
        mode = 'train'
        self.model.train()
        loss, top1_acc, top5_acc, img_paths, y_trues, y_preds, y_dists = \
            self.one_epoch(mode=mode, cur_epoch=cur_epoch)
        if self.lr_scheduler:
            self.lr_scheduler.step()
        # for logging ->
        if self.store_loss_acc_log:
            self.__update_loss_acc_state(mode, cur_epoch, loss, top1_acc, top5_acc)
            if self.tb_writer:
                self.tb_writer.add_scalar('train-loss', loss, cur_epoch)
                self.tb_writer.add_scalar('train-top1-acc', top1_acc, cur_epoch)
                self.tb_writer.add_scalar('train-top5-acc', top5_acc, cur_epoch)
        if self.store_logits:
            self.__write_csv_logits(mode, cur_epoch, img_paths, y_preds, y_dists)

    def valid(self, cur_epoch):
        mode = 'valid'
        self.model.eval()
        with torch.no_grad():
            loss, top1_acc, top5_acc, img_paths, y_trues, y_preds, y_dists = \
                self.one_epoch(mode=mode, cur_epoch=cur_epoch)
        is_best_valid = self.__update_best_valid_acc_state(top1_acc, top5_acc)
        # for logging ->
        if self.store_weights:
            self.best_model_state_dict = self.model.state_dict()
        if self.store_loss_acc_log:
            self.__update_loss_acc_state(mode, cur_epoch, loss, top1_acc, top5_acc)
            self.__write_csv_log_loss_acc()
            if self.tb_writer:
                self.tb_writer.add_scalar('valid-loss', loss, cur_epoch)
                self.tb_writer.add_scalar('valid-top1-acc', top1_acc, cur_epoch)
                self.tb_writer.add_scalar('valid-top5-acc', top5_acc, cur_epoch)
        if self.store_logits:
            self.__write_csv_logits(mode, cur_epoch, img_paths, y_preds, y_dists)
        if self.store_confusion_matrix:
            if is_best_valid:
                self.__write_confusion_matrix(mode, cur_epoch, y_preds, y_trues)

    def test(self):
        mode = 'test'
        self.model.eval()
        with torch.no_grad():
            _, top1_acc, top5_acc, img_paths, y_trues, y_preds, y_dists = \
                self.one_epoch(mode=mode, cur_epoch=-1)
        print(f'➜ top1_acc: {top1_acc: .2f}%, top5_acc: {top5_acc: .2f}%')
        # for logging ->
        if self.store_logits:
            self.__write_csv_logits(mode, -1, img_paths, y_preds, y_dists)
        if self.store_confusion_matrix:
            self.__write_confusion_matrix(mode, -1, y_preds, y_trues)

    def store_model(self):
        torch.save(self.best_model_state_dict, self.best_model_state_path)

    @classmethod
    def __get_final_classification_results_and_topk_acc__(cls, out, gt, top_k=(1, 5)):
        _, prediction = out.topk(k=max(top_k), dim=1, largest=True, sorted=True)
        prediction = prediction.t()
        correct = prediction.eq(gt.view(1, -1).expand_as(prediction))
        top_k_acc_list = [correct[:k].reshape(-1).float().sum(0, keepdim=True) for k in top_k]
        _, top_1_prediction = out.topk(k=1, dim=1, largest=True, sorted=True)
        return top_1_prediction, top_k_acc_list  # sum of correct predictions (top_1, top_k)

    @classmethod
    def __accumulate_predictions(cls, img_path, img_paths, y, y_dist, y_dists, y_pred, y_preds, y_trues):
        img_paths.extend(img_path)
        y_trues.extend(y.data.cpu().numpy())
        y_preds.extend(torch.flatten(y_pred).tolist())
        y_dists.extend([logit.tolist() for logit in y_dist.cpu().detach().numpy()])

    @classmethod
    def __update_all_meters(cls, loss, meter, top1_acc, top5_acc, k):
        meter['top1_acc'].update(top1_acc.item(), k=k)
        meter['top5_acc'].update(top5_acc.item(), k=k)
        if loss:
            meter['loss'].update(loss.item(), k=k)

    def __print_tqdm_log(self, cur_epoch, meter, mode, tqdm_loader):
        if mode in 'train':
            log_msg = f"Loss: {meter['loss'].avg:.3f} " \
                      f"| Acc: (top1) {meter['top1_acc'].avg * 100.:.2f}% " \
                      f"(top5) {meter['top5_acc'].avg * 100.:.2f}% " + \
                      (f"[top1-best-val: {self.best_valid_acc_state['top1_acc']:.2f}%]"
                       if cur_epoch > 0 else '')
        elif mode in 'valid':
            log_msg = f"Loss: {meter['loss'].avg:.3f} " \
                      f"| Acc: (top1) {meter['top1_acc'].avg * 100.:.2f}% " \
                      f"(top5) {meter['top5_acc'].avg * 100.:.2f}% "
        else:  # mode == 'test'
            log_msg = f"Acc: (top1) {meter['top1_acc'].avg * 100.:.2f}% " \
                      f"(top5) {meter['top5_acc'].avg * 100.:.2f}%"
        tqdm_loader.set_description(f'{mode.upper()} | {cur_epoch + 1:>5d} | {log_msg}')

    def __optimize_model(self, cur_epoch, loss):
        # lr-warmup
        if self.lr_warmup_scheduler and cur_epoch < self.lr_warmup_epochs:
            self.lr_warmup_scheduler.step()
        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def __update_best_valid_acc_state(self, top1_acc, top5_acc):
        if top1_acc > self.best_valid_acc_state['top1_acc'] or \
                (top1_acc == self.best_valid_acc_state['top1_acc'] and
                 top5_acc > self.best_valid_acc_state['top5_acc']):
            self.best_valid_acc_state['top1_acc'] = top1_acc
            self.best_valid_acc_state['top5_acc'] = top5_acc
            return True
        return False

    def __update_loss_acc_state(self, mode, epoch, loss, top1_acc, top5_acc):
        self.loss_acc_state['epoch'] = epoch
        self.loss_acc_state[f'{mode}_loss'] = loss
        self.loss_acc_state[f'{mode}_top1_acc'] = top1_acc
        self.loss_acc_state[f'{mode}_top5_acc'] = top5_acc

    def __write_csv_log_loss_acc(self):
        with open(self.log_loss_acc_csv_path, 'a') as f:
            self.log_loss_acc_csv_writer.writerow(self.loss_acc_state)
            f.flush()

    def __write_csv_logits(self, mode, cur_epoch, img_paths, classification_results, output_distributions):
        # assert len(img_paths) == len(classification_results) == len(output_distributions)
        zips = zip(img_paths, classification_results, output_distributions)
        sep = os.sep  # '/': linux, '\': windows
        for img_path, classification_result, output_distribution in zips:
            class_name, file_name = img_path.split(sep)[-2], img_path.split(sep)[-1]
            root_path = f'{self.logits_root_path}/{mode}/{class_name}/{file_name}'
            csv_path = f'{root_path}/logits.csv'
            if not os.path.isdir(root_path):
                os.makedirs(root_path, exist_ok=True)
                csv_writer = csv.DictWriter(open(csv_path, 'w', newline=NEWLINE),
                                            fieldnames=['epoch', 'output_distribution', 'classification_result'])
                csv_writer.writeheader()
                self.logits_csv_writers[f'{mode}/{class_name}/{file_name}'] = csv_writer
            with open(csv_path, 'a') as f:
                self.logits_csv_writers[f'{mode}/{class_name}/{file_name}'].writerow({
                    'epoch': cur_epoch,
                    'output_distribution': output_distribution,
                    'classification_result': classification_result
                })
                f.flush()

    def __write_confusion_matrix(self, mode, cur_epoch, y_preds, y_trues):
        class_names = self.loader[mode].dataset.class_names  # only valid when using src/dataset.py
        plot_cf_matrix = util.create_confusion_matrix(y_trues, y_preds,
                                                      num_of_classes=len(class_names),
                                                      class_names=class_names,
                                                      title=f"best-val "
                                                            f"[top1] {self.best_valid_acc_state['top1_acc']: .2f}% "
                                                            f"[top5] {self.best_valid_acc_state['top5_acc']: .2f}%")
        if self.tb_writer and mode != 'test':
            self.tb_writer.add_figure('Confusion Matrix', plot_cf_matrix, cur_epoch)
        file_name = f'{mode}-epoch-{cur_epoch}.svg' if cur_epoch > -1 else f'{mode}.svg'
        plot_cf_matrix.savefig(f'{self.confusion_matrix_root_path}/{file_name}', dpi=600)
