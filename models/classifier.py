from typing import Optional

import torch
import torchvision

from .model import Model


class TorchvisionClassifier(Model):
    """
    Wrapper for torchvision classifiers.
    """

    def __init__(self,
                 model: str = 'resnet18',
                 out_features: int = 2, pretrained=False, fc_attr=None,
                 logger='accuracy', loss='ce', loss_params=None, optim='adam',
                 optim_params=dict(lr=1e-3, weight_decay=1e-5),
                 scheduler=None,
                 scheduler_params=None,
                 ):

        self.save_hyperparameters()

        if not fc_attr:
            if 'resnet' in model:
                fc_attr = 'fc'
            elif any([n in model for n in ('densenet', 'inception', 'vgg')]):
                fc_attr = 'classifier'
            else:
                raise ValueError(f'Model {model} not supported without explicit value for {fc_attr}')

        model = torchvision.models.__dict__[model](pretrained=pretrained)
        model.__dict__['_modules'][fc_attr] \
            = torch.nn.Linear(model.__dict__['_modules'][fc_attr].in_features, out_features)

        super().__init__(model=model, logger=logger, loss=loss, loss_params=loss_params, optim=optim,
                         optim_params=optim_params,
                         scheduler=scheduler, scheduler_params=scheduler_params)


class MLP(Model):

    def __init__(self, input_size: int,
                 hidden_layers: list[int],
                 output_size: int,
                 dropout: Optional[float] = 0.5,
                 logger='auc', loss='ce',
                 loss_params=None, optim='adam',
                 optim_params=dict(lr=1e-3, weight_decay=0),
                 scheduler=None,
                 scheduler_params=None,
                 legacy=True,
                 ):
        self.save_hyperparameters()
        layers = [torch.nn.Linear(input_size, hidden_layers[0])]
        if not legacy:
            layers.append(torch.nn.ReLU())
        for i in range(1, len(hidden_layers)):
            layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(torch.nn.ReLU(inplace=True))

        if dropout is not None:
            layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Linear(hidden_layers[-1], output_size))
        model = torch.nn.Sequential(*layers)
        logger_params = dict(num_classes=output_size) if logger == 'auc' else None
        super().__init__(model=model, logger=logger, loss=loss, loss_params=loss_params, optim=optim,
                         optim_params=optim_params, logger_params=logger_params,
                         scheduler=scheduler, scheduler_params=scheduler_params)

