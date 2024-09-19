import torch


class ApproxOrderMSE(torch.nn.Module):
    def __init__(self, temperature: float = 1):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != -100
        approx_ranks = self.get_approx_ranks(scores, self.temperature, mask)
        ranks = torch.argsort(torch.where(mask, targets, float("inf"))) + 1
        mse = torch.nn.functional.mse_loss(approx_ranks, ranks.to(approx_ranks), reduction="none")
        masked_mse = mse * mask
        loss = (masked_mse.sum(-1) / mask.sum(-1)).mean()
        return loss

    @staticmethod
    def get_approx_ranks(scores: torch.Tensor, temperature: float, mask: torch.Tensor) -> torch.Tensor:
        score_diff = scores[:, None] - scores[..., None]
        expanded_mask = mask[:, None] & mask[..., None]
        normalized_score_diff = torch.sigmoid(score_diff / temperature)
        # set diagonal to 0
        normalized_score_diff = normalized_score_diff * (1 - torch.eye(scores.shape[1], device=scores.device))
        normalized_score_diff[~expanded_mask] = 0
        approx_ranks = normalized_score_diff.sum(-1) + 1
        approx_ranks[~mask] = 0
        return approx_ranks
