from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BarlowTwins(nn.Module):
    """Computes the "Barlow Twins" objective as proposed in "Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction".
    """

    def __init__(self, lmbda: float, num_features: int) -> None:
        """
        Args:
            lmbda (float): This hyper parameter is the square-root of "lambda" from the original paper.
        """
        super().__init__()
        self._alpha = sqrt(lmbda)
        self.batchnorm = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, features1: Tensor, features2: Tensor) -> Tensor:
        features1 = features1.reshape(-1, features1.shape[-1])
        features2 = features2.reshape(-1, features2.shape[-1])
        N, D = features1.shape  # N is batchsize*num_features and D is the embedding dimensionality
        assert list(features2.shape) == [
            N,
            D,
        ], f"Both embeddings must have the same shape, got {features1.shape} and {features2.shape}"
        normed1 = self.batchnorm(features1)
        normed2 = self.batchnorm(features2)
        crosscorr = normed1.T @ normed2 / N

        assert list(crosscorr.shape) == [D, D]
        # 1 a a ... a
        # a 1 a ... a
        # a a 1     a
        # . .       .
        # a a . 1 a a
        # a a . a 1 a
        # a a . a a 1
        weights = (1 - torch.eye(D, device=crosscorr.device)) * self._alpha + torch.eye(D, device=crosscorr.device)
        return F.mse_loss(crosscorr * weights, torch.eye(D, device=crosscorr.device), reduction="sum")
