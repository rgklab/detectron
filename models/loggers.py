from abc import ABC, abstractmethod
from typing import List

import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
import torch
import torch.nn.functional as F


class Logger(torch.nn.Module):
    """
    Abstract class for logging
    """

    def __init__(self, log_fn):
        super().__init__()
        self.log = log_fn

    @abstractmethod
    def on_train(self, y_hat, y, loss=None):
        pass

    @abstractmethod
    def on_val(self, y_hat, y, loss=None):
        pass

    @abstractmethod
    def after_train(self):
        pass

    @abstractmethod
    def after_val(self):
        pass


class Compose(Logger):
    def __init__(self, log_fn, loggers: List[Logger]):
        super().__init__(log_fn)
        self.loggers = loggers

    def on_val(self, y_hat, y, loss=None):
        for logger in self.loggers:
            logger.on_val(y_hat, y, loss=loss)

    def after_train(self):
        for logger in self.loggers:
            logger.after_train()

    def after_val(self):
        for logger in self.loggers:
            logger.after_val()

    def on_train(self, y_hat, y, loss=None):
        for logger in self.loggers:
            logger.on_train(y_hat, y, loss=loss)


class AccuracyLogger(Logger):
    """
    Logger for accuracy
    """

    def __init__(self, log_fn, log_on_step: bool = True, num_classes=10):
        super().__init__(log_fn)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.log_on_step = log_on_step

    def on_train(self, y_hat, y, loss=None):
        a = self.train_acc(y_hat, y)
        if self.log_on_step:
            self.log('train_acc_step', a)
            if loss:
                self.log('train_loss_step', loss)

    def on_val(self, y_hat, y, loss=None):
        a = self.val_acc(y_hat, y)
        if self.log_on_step:
            self.log('val_acc_step', a)
            if loss:
                self.log('val_loss_step', loss)

    def after_train(self):
        a = self.train_acc.compute()
        self.train_acc.reset()
        self.log('train_acc', a)

    def after_val(self):
        a = self.val_acc.compute()
        self.val_acc.reset()
        self.log('val_acc', a)


class AUCLogger(Logger):
    """
    Logger for accuracy
    """

    def __init__(self, log_fn, log_on_step: bool = True, num_classes: int = 2):
        super().__init__(log_fn)
        self.train_auc = AUROC(num_classes=num_classes)
        self.val_auc = AUROC(num_classes=num_classes)
        self.log_on_step = log_on_step

    def on_train(self, y_hat, y, loss=None):
        a = self.train_auc(F.softmax(y_hat, dim=1), y)
        if self.log_on_step:
            self.log('train_auc_step', a)
            if loss:
                self.log('train_loss_step', loss)

    def on_val(self, y_hat, y, loss=None):
        a = self.val_auc(F.softmax(y_hat, dim=1), y)
        if self.log_on_step:
            self.log('val_auc_step', a)
            if loss:
                self.log('val_loss_step', loss)

    def after_train(self):
        a = self.train_auc.compute()
        self.train_auc.reset()
        self.log('train_auc', a)

    def after_val(self):
        a = self.val_auc.compute()
        self.val_auc.reset()
        self.log('val_auc', a)


class LogitLogger(Logger):
    def __init__(self, log_fn):
        super().__init__(log_fn)
        self.logits = []

    def on_train(self, y_hat, y, loss=None):
        pass

    def on_val(self, y_hat, y, loss=None):
        self.logits.append(y_hat.cpu())

    def after_train(self):
        pass

    def after_val(self):
        pass
