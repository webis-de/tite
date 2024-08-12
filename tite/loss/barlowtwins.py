from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BarlowTwins(nn.Module):
    """Computes the "Barlow Twins" objective as proposed in "Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction".
    """

    def __init__(self, lmbda: float, embeddim: int) -> None:
        """
        Args:
            lmbda (float): This hyper parameter is also called "lambda" in the original paper.
            embeddim (int): The embedding dimensionality
        """
        super().__init__()
        self._alpha = sqrt(lmbda)
        self.batchnorm = nn.BatchNorm1d(embeddim, affine=False)
        self.register_buffer(
            "weights", torch.where(torch.eye(embeddim, dtype=torch.bool), 1, self._alpha), persistent=False
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, features1: Tensor, features2: Tensor) -> Tensor:
        features1 = features1.view(-1, features1.shape[-1])
        features2 = features2.view(-1, features2.shape[-1])
        N, D = features1.shape  # N is batchsize*num features and D is embeddim
        assert list(features2.shape) == [
            N,
            D,
        ], f"Both embeddings must have the same shape, got {features1.shape} and {features2.shape}"
        normed1 = self.batchnorm(features1)
        normed2 = self.batchnorm(features2)
        crosscorr = normed1.T @ normed2 / N

        assert crosscorr.shape == self.weights.shape
        return F.mse_loss(crosscorr * self.weights, torch.eye(D, device=crosscorr.device), reduction="sum") / D


class ProjectedBarlowTwins(nn.Module):

    def __init__(self, lmbda: float, embeddim: int, sizes: tuple[int, ...] = (8192,) * 3) -> None:
        """
        Args:
            lmbda (float): This hyper parameter is also called "lambda" in the original paper.
            embeddim (int): The embedding dimensionality
            sizes (tuple[int]): The intermediate dimensions for the projector
        """
        super().__init__()
        # Projector
        sizes = [embeddim] + list(sizes)
        layers = []
        for i in range(len(sizes) - 2):
            layers.extend([nn.Linear(sizes[i], sizes[i + 1], bias=False), nn.BatchNorm1d(sizes[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projector = nn.Sequential(*layers)
        # BarlowTwins
        self.bt = BarlowTwins(lmbda=lmbda, embeddim=sizes[-1])

    def forward(self, features1: Tensor, features2: Tensor):
        features1 = features1.view(-1, features1.shape[-1])
        features2 = features2.view(-1, features2.shape[-1])
        return self.bt(self.projector(features1), self.projector(features2))
