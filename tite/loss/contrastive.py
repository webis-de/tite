import torch

from .loss import LossFunction


class ContrastiveInBatchSimilarityLoss(LossFunction):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, embx: torch.Tensor, emby: torch.Tensor) -> torch.Tensor:
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
