import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.linalg import matrix_norm
from torch.nn import Module


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


class ContrastiveInBatchSimilarity(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, embx, emby):
        assert embx.shape == emby.shape
        batch_size = embx.shape[0]
        embedding_dim = embx.shape[-1]
        bool_mask = (1 - torch.eye(batch_size)).bool()
        idcs = torch.nonzero(bool_mask, as_tuple=True)
        neg_samples = torch.cat(
            [
                embx[idcs[1]].view(batch_size, batch_size - 1, embedding_dim),
                emby[idcs[1]].view(batch_size, batch_size - 1, embedding_dim),
            ],
            dim=1,
        )

        pos_sim = torch.matmul(embx, emby.transpose(-1, -2)).squeeze(1)
        neg_sim_x = torch.matmul(embx, neg_samples.transpose(-1, -2)).squeeze(1)
        neg_sim_y = torch.matmul(emby, neg_samples.transpose(-1, -2)).squeeze(1)
        sim_x = torch.cat([pos_sim, neg_sim_x], dim=1)
        sim_y = torch.cat([pos_sim, neg_sim_y], dim=1)
        targets = torch.zeros(batch_size, dtype=torch.long, device=embx.device)
        loss_x = torch.nn.functional.cross_entropy(sim_x, targets)
        loss_y = torch.nn.functional.cross_entropy(sim_y, targets)
        return (loss_x + loss_y) / 2


class MLMCrossEntropy(Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits, labels):
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        return torch.nn.functional.cross_entropy(logits, labels)


class OrigMLMCrossEntropy(Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits, labels):
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        return torch.nn.functional.cross_entropy(logits, labels)


class BOWBinaryCrossEntropy(Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits, labels):
        logits = logits.view(-1)
        labels = labels.view(-1)
        mask = labels == -100
        logits = logits[~mask]
        labels = labels[~mask]
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


class StandardDivRegularization(Module):

    def forward(self, embx: Tensor) -> Tensor:
        stds = embx.std(0)
        return torch.nn.functional.mse_loss(stds, torch.tensor(1.0, device=stds.device))


def mmcr(embeddings: Tensor) -> Tensor:
    r"""Implements the "Maximum Manifold Capacity Representation" loss.

    Let :math:`X\in\mathbb{R}^{K\times P\times D}` denote the normalized embeddings obtained from embedding :math:`K`
    transformations of :math:`P` data points. Here, :math:`D` denotes the embedding dimensionality. The maximum
    manifold representations loss is defined as the nuclear norm of the :math:`P\times D` matrix made up from the
    centers of each data point's embeddings:

    .. math::
        \mathcal{L}_\textrm{MMCR} := -\lVert C\rVert_\ast, \textrm{ with }
        C := (\frac{1}{K} \sum_{1\leq k\leq K} x_{k,p,d})_{1\leq p\leq P; 1\leq d\leq d}

    Args:
        embeddings (Tensor): A 3D tensor of shape (K, P, D), where K is the number of perturbations, P is the number of embedded data points and D is the embedding dimensionality.

    Returns:
        Tensor: The mmcr loss
    """
    K, P, D = embeddings.shape
    normed = F.normalize(embeddings, dim=-1)
    centers = normed.mean(0)
    assert list(centers.shape) == [P, D]
    return -matrix_norm(centers, ord="nuc")


class MMCRLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, embeddings: Tensor) -> Tensor:
        return mmcr(embeddings)


class ApproxOrderMSE(torch.nn.Module):
    def __init__(self, temperature: float = 1):
        super().__init__()
        self.temperature = temperature

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, scores: torch.FloatTensor, targets: torch.LongTensor) -> torch.Tensor:
        mask = targets != -100
        approx_ranks = ApproxOrderMSE.get_approx_ranks(scores, self.temperature, mask)
        ranks = torch.argsort(torch.where(mask, targets, float("inf"))) + 1
        mse = torch.nn.functional.mse_loss(approx_ranks, ranks.to(approx_ranks), reduction="none")
        masked_mse = mse * mask
        loss = (masked_mse.sum(-1) / mask.sum(-1)).mean()
        return loss

    @staticmethod
    def get_approx_ranks(scores: torch.FloatTensor, temperature: float, mask: torch.BoolTensor) -> torch.Tensor:
        score_diff = scores[:, None] - scores[..., None]
        expanded_mask = mask[:, None] & mask[..., None]
        normalized_score_diff = torch.sigmoid(score_diff / temperature)
        approx_ranks = normalized_score_diff.masked_fill(~expanded_mask, 0).sum(-1) + 0.5
        approx_ranks = approx_ranks.masked_fill(~mask, 0)
        return approx_ranks
