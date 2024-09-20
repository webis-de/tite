import torch


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
        normalized_score_diff[~expanded_mask] = 0
        approx_ranks = normalized_score_diff.sum(-1) + 0.5
        approx_ranks[~mask] = 0
        return approx_ranks
