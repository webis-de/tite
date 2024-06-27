import torch
import torch.nn as nn
from torch import Tensor


class BarlowTwins(nn.Module):
    """Computes the "Barlow Twins" objective as proposed in "Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction".
    """

    def __init__(self, alpha: float, num_features: int) -> None:
        """
        Args:
            alpha (float): In the paper, this hyper parameter is called "lambda". It should be positive.
        """
        super().__init__()
        self._alpha = alpha
        self.batchnorm = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, features1: Tensor, features2: Tensor) -> Tensor:
        N, D = features1.shape
        assert list(features2.shape) == [
            N,
            D,
        ], f"Both embeddings must have the same shape, got {features1.shape} and {features2.shape}"
        normed1 = self.batchnorm(features1)
        normed2 = self.batchnorm(features2)
        crosscorr = normed1.T @ normed2 / N

        assert list(crosscorr.shape) == [D, D]
        obj = (crosscorr - torch.eye(D)).pow(2)
        obj[~torch.eye(D, dtype=torch.bool)] *= self._alpha
        return obj.sum()
