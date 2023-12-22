from typing import Any, Optional

import pytorch_lightning as pl
import torch

from models import configs


class Model(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module, logger: str = None,
                 logger_params: dict[str, Any] = None,
                 loss: str = None, loss_params: dict[str, Any] = None, optim: str = None,
                 optim_params: dict[str, Any] = None,
                 scheduler: str = None, scheduler_params: dict[str, Any] = None):
        super().__init__()
        self.model = model
        self.optimizer = configs.get_optim(optim, optim_params)
        self.scheduler = configs.get_scheduler(scheduler, scheduler_params)
        self.loss_fn = configs.get_loss(loss, loss_params)
        self._logger = configs.get_logger(logger)(self.log, **(logger_params or {}))

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch[0]
        y = batch[1]

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self._logger.on_train(y_hat, y, loss=loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self._logger.on_val(y_hat, y, loss=loss)

    def on_train_epoch_end(self, unused: Optional[Any] = None) -> None:
        self._logger.after_train()

    def on_validation_epoch_end(self) -> None:
        self._logger.after_val()

    def configure_optimizers(self):
        optim = self.optimizer(self.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optim)
            return [optim], [scheduler]
        return optim
