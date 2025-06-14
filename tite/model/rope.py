# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None
from torch import Tensor, nn

from .pool import PackedMetaData

# https://pytorch.org/torchtune/0.2/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings


class EagerRotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(self, dim: int, base: int = 10_000, interleaved: bool = True) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be divisible by 2"
        self.dim = dim
        self.base = base
        if not interleaved:
            raise NotImplementedError("Non-interleaved RoPE is not supported yet.")
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self) -> None:
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)

    def _update_rope_cache(self, seq_len: int) -> None:
        cache = getattr(self, "cache", None)
        if cache is not None and cache.shape[0] >= seq_len:
            return
        seq_len = 1 << (1000 - 1).bit_length()  # round up to next power of 2
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=self.theta.device)

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, packed_meta_data: PackedMetaData) -> Tensor:
        self._update_rope_cache(packed_meta_data.max_seq_len)
        rope_cache = self.cache[packed_meta_data.position_idcs].view(x.shape[0], 1, self.dim // 2, 2)

        xshaped = x.view(*x.shape[:-1], self.dim // 2, 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        return x_out.view_as(x).to(x)


class TritonRotaryPositionalEmbeddings(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        base: float = 10_000,
        pos_idx_in_fp32: bool = True,
        interleaved: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device: torch.device | None) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        seq_len = 1 << (1000 - 1).bit_length()  # round up to next power of 2
        if (
            seq_len > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seq_len
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seq_len, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, x: torch.Tensor, packed_meta_data: PackedMetaData) -> torch.Tensor:
        self._update_cos_sin_cache(packed_meta_data.max_seq_len, device=x.device, dtype=x.dtype)

        x = apply_rotary_emb(
            x,
            self._cos_cached,
            self._sin_cached,
            inplace=True,
            interleaved=self.interleaved,
            cu_seqlens=packed_meta_data.cu_seq_lens,
            max_seqlen=packed_meta_data.max_seq_len,
        )
        return x
