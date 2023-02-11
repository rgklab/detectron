import warnings
from glob import glob
from os.path import join
from typing import Optional, Callable

import attr
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from nn.dce_loss import DCELoss


@attr.s(auto_attribs=True)
class DetectronStruct:
    rejection_count: int = None
    base_accuracy: float = None,
    detector_accuracy: float = None
    accepted_accuracy: float = None
    rejection_rate: float = None
    logits: torch.Tensor = None
    rejection_mask: torch.Tensor = None

    def to_dict(self, minimal: bool = False):
        if minimal:
            return {
                'base_accuracy': self.base_accuracy,
                'detector_accuracy': self.detector_accuracy,
                'rejection_rate': self.rejection_rate,
                'accepted_accuracy': self.accepted_accuracy,
            }
        return attr.asdict(self)


class DetectronModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, alpha=None, optim_func: Callable = None,
                 num_classes=10):
        super().__init__()
        self.model = model
        self.loss = DCELoss(alpha=alpha)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.q_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.q_reject = torch.zeros(1, dtype=torch.bool)
        self.q_logits = torch.zeros(1, dtype=torch.float)
        self.test_struct = DetectronStruct()
        self.optim_func = optim_func

    def set_alpha(self, alpha):
        self.loss.alpha = alpha

    def training_step(self, batch, batch_idx):
        x, y_hat, y, mask = batch
        logits = self.model(x)
        loss = self.loss(logits=logits, labels=y_hat, mask=mask)
        return loss

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_id):
        x, yhat, y, mask = batch
        logits = self.model(x)
        pred = logits.argmax(dim=1)
        reject_mask = ~torch.eq(pred, yhat)
        base_correct = torch.eq(yhat, y)
        detector_correct = torch.eq(pred, y)
        return dict(logits=logits, reject_mask=reject_mask, base_correct=base_correct,
                    detector_correct=detector_correct)

    def test_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs], dim=0)
        reject_mask = torch.cat([x['reject_mask'] for x in outputs], dim=0)
        base_correct = torch.cat([x['base_correct'] for x in outputs], dim=0)
        detector_correct = torch.cat([x['detector_correct'] for x in outputs], dim=0)
        self.test_struct = DetectronStruct(
            rejection_mask=reject_mask.cpu(),
            logits=logits.cpu(),
            base_accuracy=base_correct.float().mean().item(),
            rejection_count=reject_mask.float().sum().item(),
            rejection_rate=reject_mask.float().mean().item(),
            accepted_accuracy=base_correct[~reject_mask].float().mean().item(),
            detector_accuracy=detector_correct.float().mean().item(),
        )

    def validation_step(self, batch, batch_idx):
        x, y_hat, y, mask = batch
        logits = self.model(x)
        self.val_acc(logits.argmax(dim=1), y)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute())
        self.val_acc.reset()

    def configure_optimizers(self):
        if self.optim_func is None:
            return torch.optim.Adam(self.model.parameters())
        return self.optim_func(self.model.parameters())


class DetectronEnsemble(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, detectors: Optional[torch.nn.ModuleList] = None):
        super().__init__()
        if detectors is None:
            self.ensemble = torch.nn.ModuleList([base_model])
        else:
            self.ensemble = torch.nn.ModuleList([base_model, *detectors])

    def add_detector(self, detector: torch.nn.Module):
        self.ensemble.append(detector)

    @staticmethod
    def load_from_checkpoint(base_model: torch.nn.Module,
                             checkpoint_directory: str,
                             model_cls: pl.LightningModule,
                             sorting_func: Optional[Callable[[str], int]] = None) -> 'DetectronEnsemble':
        checkpoints = glob(join(checkpoint_directory, '*.ckpt'))
        if len(checkpoints) == 0:
            warnings.warn(f'No checkpoints found in {checkpoint_directory}, using base model only')
            return DetectronEnsemble(base_model, None)

        if sorting_func is not None:
            checkpoints.sort(key=sorting_func)

        return DetectronEnsemble(
            base_model,
            torch.nn.ModuleList(
                [model_cls.load_from_checkpoint(c) for c in checkpoints]
            )
        )

    def forward(self, x):
        return torch.stack([model(x) for model in self.ensemble], dim=1)


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0
        assert mode in ['min', 'max']
        self.mode = mode

    def update(self, metric):
        if self.best is None:
            self.best = metric
            return False
        if self.mode == 'min':
            if metric < self.best - self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience
        elif self.mode == 'max':
            if metric > self.best + self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience
