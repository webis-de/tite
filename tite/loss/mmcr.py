import torch

from .loss import LossFunction


class MMCRLoss(LossFunction):
    """Implements the "Maximum Manifold Capacity Representation" loss.

    Let :math:`X\in\mathbb{R}^{K\times P\times D}` denote the normalized embeddings obtained from embedding :math:`K`
    transformations of :math:`P` data points. Here, :math:`D` denotes the embedding dimensionality. The maximum
    manifold representations loss is defined as the nuclear norm of the :math:`P\times D` matrix made up from the
    centers of each data point's embeddings:

    .. math::
        \mathcal{L}_\textrm{MMCR} := -\lVert C\rVert_\ast, \textrm{ with }
        C := (\frac{1}{K} \sum_{1\leq k\leq K} x_{k,p,d})_{1\leq p\leq P; 1\leq d\leq d}

    Args:
        embeddings (Tensor): A 3D tensor of shape (K, P, D), where K is the number of perturbations, P is the number of
            embedded data points and D is the embedding dimensionality.

    Returns:
        Tensor: The mmcr loss
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        K, P, D = embeddings.shape
        normed = F.normalize(embeddings, dim=-1)
        centers = normed.mean(0)
        assert list(centers.shape) == [P, D]
        return -torch.linalg.matrix_norm(centers, ord="nuc")
