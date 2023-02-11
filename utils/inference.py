import pytorch_lightning as pl
import torch


class LabelCollector(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.predicted_labels = []
        self.true_labels = []

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat: torch.Tensor = self.model(x).argmax(dim=1)
        self.predicted_labels.extend(y_hat.cpu().tolist())
        self.true_labels.extend(y.cpu().tolist())

    def compute_accuracy(self):
        return torch.eq(torch.tensor(self.predicted_labels), torch.tensor(self.true_labels)).float().mean().item()

    def get_labels(self, mode='predicted'):
        if mode == 'predicted':
            return self.predicted_labels
        elif mode == 'true':
            return self.true_labels
        else:
            raise ValueError(f"Invalid mode: {mode}")
