import torch
from torch import Tensor


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(torch.nn.Module):
    """Computes the "Barlow Twins" objective as proposed in "Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction".
    """

    # https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py#L207

    def __init__(self, lmbda: float, embeddim: int) -> None:
        """
        Args:
            lmbda (float): This hyper parameter is also called "lambda" in the original paper.
            embeddim (int): The embedding dimensionality
        """
        super().__init__()
        self.lmbda = lmbda
        self.batch_norm = torch.nn.BatchNorm1d(embeddim, affine=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, features1: Tensor, features2: Tensor) -> Tensor:
        features1 = features1.view(-1, features1.shape[-1])
        features2 = features2.view(-1, features2.shape[-1])
        # empirical cross-correlation matrix
        c = (self.batch_norm(features1).T @ self.batch_norm(features2)) / features1.shape[0]

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lmbda * off_diag
        return loss


class ProjectedBarlowTwins(torch.nn.Module):

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
            layers.extend(
                [
                    torch.nn.Linear(sizes[i], sizes[i + 1], bias=False),
                    torch.nn.BatchNorm1d(sizes[i + 1]),
                    torch.nn.ReLU(),
                ]
            )
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
        self.projector = torch.nn.Sequential(*layers)
        # BarlowTwins
        self.bt = BarlowTwins(lmbda=lmbda, embeddim=sizes[-1])

    def forward(self, features1: Tensor, features2: Tensor):
        features1 = features1.view(-1, features1.shape[-1])
        features2 = features2.view(-1, features2.shape[-1])
        return self.bt(self.projector(features1), self.projector(features2))
