from typing import Optional

import torch
import torch.nn.functional as F


class DCELoss(torch.nn.Module):
    def __init__(self, weight=None, use_random_vectors=False, alpha=None):
        super(DCELoss, self).__init__()
        self.weight = weight
        self.use_random_vectors = use_random_vectors
        self.alpha = alpha

    def forward(self, logits, labels, mask):
        return dce_loss(logits, labels,
                        weight=self.weight,
                        mask=mask,
                        use_random_vectors=self.use_random_vectors,
                        alpha=self.alpha)


def dce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, alpha: Optional[float] = None,
             use_random_vectors=False, weight=None) -> torch.Tensor:
    """

    :param logits: (batch_size, num_classes) tensor of logits
    :param labels: (batch_size,) tensor of labels
    :param mask: (batch_size,) mask
    :param alpha: (float) weight of q samples
    :param use_random_vectors: (bool) whether to use random vectors for negative labels, default=False
    :param weight:  (torch.Tensor) weight for each sample_data, default=None do not apply weighting
    :return: (tensor, float) the disagreement cross entropy loss
    """
    if mask.all():
        # if all labels are positive, then use the standard cross entropy loss
        return F.cross_entropy(logits, labels)

    if alpha is None:
        alpha = 1 / (1 + (~mask).float().sum())

    num_classes = logits.shape[1]

    q_logits, q_labels = logits[~mask], labels[~mask]
    if use_random_vectors:
        # noinspection PyTypeChecker,PyUnresolvedReferences
        p = - torch.log(torch.rand(device=q_labels.device, size=(len(q_labels), num_classes)))
        p *= (1. - F.one_hot(q_labels, num_classes=num_classes))
        p /= torch.sum(p)
        ce_n = -(p * q_logits).sum(1) + torch.logsumexp(q_logits, dim=1)

    else:
        zero_hot = 1. - F.one_hot(q_labels, num_classes=num_classes)
        ce_n = -(q_logits * zero_hot).sum(dim=1) / (num_classes - 1) + torch.logsumexp(q_logits, dim=1)

    if torch.isinf(ce_n).any() or torch.isnan(ce_n).any():
        raise RuntimeError('NaN or Infinite loss encountered for ce-q')

    if (~mask).all():
        return (ce_n * alpha).mean()

    p_logits, p_labels = logits[mask], labels[mask]
    ce_p = F.cross_entropy(p_logits, p_labels, reduction='none', weight=weight)
    return torch.cat([ce_n * alpha, ce_p]).mean()
