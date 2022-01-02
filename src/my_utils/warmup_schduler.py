from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        self.total_iters = 0
        super().__init__(optimizer, last_epoch)

    def set_total_iters(self, total_iters):
        self.total_iters = total_iters

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]