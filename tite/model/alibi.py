import torch


class AlibiPositionalEmbeddings(torch.nn.Module):
    def __init__(self, num_attention_heads: int) -> None:
        super().__init__()

        x = 256 ** (1 / num_attention_heads)
        slopes = 1 / x ** (torch.arange(1, num_attention_heads + 1))
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor."
        max_len = max(mask.shape[-2], mask.shape[-1])
        x = torch.linspace(start=1, end=max_len, steps=mask.shape[-2], device=mask.device)
        y = torch.linspace(start=1, end=max_len, steps=mask.shape[-1], device=mask.device)

        relative_positions = (y[None] - x[:, None]).abs()
        bias = -(self.slopes[:, None, None] * relative_positions)

        bias = bias[None].masked_fill(~mask, torch.finfo(bias.dtype).min)
        return bias
