import warnings
from typing import Dict, Callable, Optional, Iterator

import torch
from models import loggers

loss_function_config: Dict[str, torch.nn.Module.__class__] = {
    'ce': torch.nn.CrossEntropyLoss,
}


def get_loss(loss_name: str, params=None) -> Optional[torch.nn.Module]:
    if loss_name is None:
        return None
        # warnings.warn('No loss function specified')

    loss_fn = loss_function_config.get(loss_name)

    if params is not None:
        return loss_fn(**params)
    return loss_fn()


logging_config: Dict[str, loggers.Logger.__class__] = {
    'accuracy': loggers.AccuracyLogger,
    'logits': loggers.LogitLogger,
    'auc': loggers.AUCLogger,
}


def get_logger(logger_name: str) -> loggers.Logger.__class__:
    return logging_config.get(logger_name, lambda *args, **kwargs: None)


def get_optim(optim_name: str, optim_params: Dict[str, float]) \
        -> Optional[Callable[[Iterator[torch.nn.Parameter], ], torch.optim.Optimizer]]:
    if optim_name is None:
        warnings.warn('No optimizer specified')
        return None
    elif optim_name == 'sgd':
        return lambda model_params: torch.optim.SGD(model_params, **optim_params)
    elif optim_name == 'adam':
        return lambda model_params: torch.optim.Adam(model_params, **optim_params)
    else:
        raise ValueError(f'Unknown optimizer {optim_name}')


# noinspection PyProtectedMember,PyUnresolvedReferences
def get_scheduler(scheduler_name: str, scheduler_params: Dict[str, float]) \
        -> Optional[Callable[[torch.optim.Optimizer, ], torch.optim.lr_scheduler._LRScheduler]]:
    if scheduler_name is None:
        # warnings.warn('No scheduler specified')
        return None
    elif scheduler_name == 'step':
        return lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == 'plateau':
        return lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_name == 'exponential':
        return lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_name == 'cosine':
        return lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f'Unknown scheduler {scheduler_name}')
