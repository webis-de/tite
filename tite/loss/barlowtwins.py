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
            alpha (float): This hyper parameter is the square-root of "lambda" from the original paper. It should be positive (but does not matter since we square it anyway).
        """
        super().__init__()
        self._alpha = alpha
        self.batchnorm = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, features1: Tensor, features2: Tensor) -> Tensor:
        features1 = features1.view(-1, features1.shape[-1])
        features2 = features2.view(-1, features2.shape[-1])
        N, D = features1.shape  # N is batchsize*num_features and D is the embedding dimensionality
        assert list(features2.shape) == [
            N,
            D,
        ], f"Both embeddings must have the same shape, got {features1.shape} and {features2.shape}"
        normed1 = self.batchnorm(features1)
        normed2 = self.batchnorm(features2)
        crosscorr = normed1.T @ normed2 / N

        assert list(crosscorr.shape) == [D, D]
        obj = crosscorr - torch.eye(D, device=crosscorr.device)
        obj[~torch.eye(D, dtype=torch.bool)] = obj[~torch.eye(D, dtype=torch.bool)] * self._alpha
        return obj.pow(2).sum()
