from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import List

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, warmup_epochs, initial_lr=0.0, after_scheduler=None):
        self.optimizer: Optimizer
        self.last_epoch: int
        self.base_lrs: List[int]

        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = warmup_epochs
        self.initial_lr = initial_lr
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch or self.total_epoch == 0:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        epoch_frac = float(self.last_epoch) / self.total_epoch
        if self.multiplier == 1.0:
            return [self.initial_lr + (base_lr - self.initial_lr) * epoch_frac
                    for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * epoch_frac + 1.)
                    for base_lr in self.base_lrs]

    def _step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        class _enable_get_lr_call:

             def __init__(self, o):
                 self.o = o

             def __enter__(self):
                 self.o._get_lr_called_within_step = True
                 return self

             def __exit__(self, type, value, traceback):
                 self.o._get_lr_called_within_step = False
                 return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return self._step(epoch=epoch)

