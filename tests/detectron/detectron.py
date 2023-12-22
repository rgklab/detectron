from __future__ import annotations

from typing import Callable, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint

from utils.generic import vprint
from utils.inference import LabelCollector

from utils.detectron import DetectronLoader, DetectronModule, DetectronEnsemble


def detectron_tst(p_train: Dataset,
                  p_val: Dataset,
                  p_test: Dataset,
                  q: Dataset,
                  base_model: torch.nn.Module,
                  create_detector: Callable[[], torch.nn.Module],
                  batch_size: int = 512,
                  ensemble_size=10,
                  max_epochs_per_model=4,
                  metric='accuracy',
                  tolerance=0.05,
                  init_metric_val=None,
                  gpus=[1],
                  num_workers=16,
                  verbose=True,
                  logging_dir=None,
                  **trainer_kwargs):
    N = len(q)
    print_fn = vprint(verbose)
    pseudo_labels = infer_labels(
        model=base_model,
        dataset=(p_train, p_val, p_test, q),
        gpus=gpus,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose
    )

    pq_loader = DetectronLoader(p_train=p_train, p_val=p_val,
                                p_test=p_test, q=q,
                                p_train_pseudo_labels=pseudo_labels[0],
                                p_val_pseudo_labels=pseudo_labels[1],
                                p_test_pseudo_labels=pseudo_labels[2],
                                q_pseudo_labels=pseudo_labels[3],
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=True)

    trainer = pl.Trainer(accelerator='auto', devices='auto', max_epochs=max_epochs_per_model, **trainer_kwargs)
    ensemble = DetectronEnsemble(base_model)

    # train <ensemble_size> CDCs to agree on p_train and disagree on q
    for k in range(1, ensemble_size + 1):
        alpha = 1 / (len(pq_loader.train_dataloader()) * N + 1)
        detector = DetectronModule(model=create_detector(), alpha=alpha)
        trainer.fit(detector, datamodule=pq_loader)
        result = detector.eval_q(pq_loader.train_dataloader())
        print_fn(f'Detector #{k} complete with test rejection rate of {detector.get_rejection_rate(N) * 100:.2f}%')

        pq_loader.refine(result[''], verbose=verbose)
        ensemble.add_detector(detector.model)


def infer_labels(model: torch.nn.Module, dataset: Dataset | Sequence[Dataset], batch_size: Optional[int] = None,
                 num_workers=64, gpus=[0],
                 verbose=True, return_accuracy=False):
    tr = pl.Trainer(accelerator='auto', devices='auto', max_epochs=1, enable_model_summary=False, logger=False)
    if isinstance(dataset, Dataset):
        dataset = [dataset]

    results = []
    for d in dataset:
        dl = DataLoader(d, batch_size=batch_size if batch_size else len(dataset),
                        num_workers=num_workers,
                        drop_last=False)

        lc = LabelCollector(model=model)
        tr.validate(lc, dl, verbose=False)
        if verbose:
            print(f'Inferred labels for {len(d)} samples. Accuracy: {lc.compute_accuracy():.3f}')
        results.append(lc.get_labels(mode='predicted'))
        if return_accuracy:
            results[-1] = [results[-1], (lc.compute_accuracy())]

    if len(dataset) == 1:
        if return_accuracy:
            return results[0][0], results[0][1]
        return results[0]
    return results
