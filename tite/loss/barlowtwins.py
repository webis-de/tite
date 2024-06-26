from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


def barlow_twins(features1: Tensor, features2: Tensor, alpha: float) -> Tensor:
    N, D = features1.shape
    assert list(features2.shape) == [
        N,
        D,
    ], f"Both embeddings must have the same shape, got {features1.shape} and {features2.shape}"
    normed1 = F.layer_norm(features1.T, (N,))
    normed2 = F.layer_norm(features2.T, (N,)).T
    crosscorr = normed1 @ normed2 / N
    assert list(crosscorr.shape) == [D, D]
    obj = (crosscorr - torch.eye(D)).pow(2)
    obj[~torch.eye(D, dtype=torch.bool)] *= alpha
    return obj.sum()


class BarlowTwins(nn.Module):
    """Computes the "Barlow Twins" objective as proposed in "Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction".
    """

    def __init__(self, alpha: float) -> None:
        """
        Args:
            alpha (float): In the paper, this hyper parameter is called "lambda". It should be positive.
        """
        self._alpha = alpha

    def forward(self, features1: Tensor, features2: Tensor) -> Tensor:
        return barlow_twins(features1, features2, self._alpha)
