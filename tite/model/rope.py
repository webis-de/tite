# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Literal

import torch

try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

from .pool import PackedMetaData

# https://pytorch.org/torchtune/0.2/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings


class RotaryPositionalEmbeddings(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        base: float = 10_000,
        interleaved: bool = True,
        implementation: Literal["triton", "eager"] = "triton",
    ) -> None:
        super().__init__()
        if implementation == "eager" and not interleaved:
            raise NotImplementedError("Non-interleaved RoPE is not supported yet.")
        if implementation == "triton" and apply_rotary_emb is None:
            raise ImportError(
                "Triton RoPE requires flash_attn to be installed. Please install it with `pip install flash-attn`."
            )
        if implementation == "triton":
            raise NotImplementedError(
                "Triton RoPE is not implemented yet. Please use the 'eager' implementation for now."
            )
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.implementation = implementation

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached = None
        self._sin_cached = None

    def forward(
        self,
        x: torch.Tensor,
        packed_meta_data: PackedMetaData,
        kernel_size: int | None = None,
        stride: int | None = None,
    ) -> torch.Tensor:
        # self._update_cos_sin_cached(packed_meta_data.max_seq_len)
        if self.implementation == "eager":
            return self.eager_forward(x, packed_meta_data, kernel_size, stride)
        # elif self.implementation == "triton":
        #     return self.triton_forward(x, packed_meta_data, unpooled_seq_lens)
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

    # def _update_cos_sin_cached(self, seq_len: int) -> None:
    #     if seq_len <= self._cached_seq_len:
    #         return
    #     seq_len = 1 << (seq_len - 1).bit_length()  # round up to next power of 2
    #     self._cached_seq_len = seq_len
    #     # Don't do einsum, it converts fp32 to fp16 under AMP
    #     # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    #     t = torch.arange(self._cached_seq_len, dtype=torch.float32, device=self.inv_freq.device)
    #     freqs = torch.outer(t, self.inv_freq)
    #     self._cos_cached = torch.cos(freqs)
    #     self._sin_cached = torch.sin(freqs)

    def eager_forward(
        self,
        x: torch.Tensor,
        packed_meta_data: PackedMetaData,
        kernel_size: int | None = None,
        stride: int | None = None,
    ) -> torch.Tensor:
        positions = packed_meta_data.position_idcs
        if kernel_size is not None and stride is not None:
            positions = positions.float() * stride + (kernel_size - 1) / 2.0

        cos = torch.cos(torch.outer(positions, self.inv_freq))[:, None]
        sin = torch.sin(torch.outer(positions, self.inv_freq))[:, None]

        xshaped = x.view(*x.shape[:-1], self.dim // 2, 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * cos - xshaped[..., 1] * sin,
                xshaped[..., 1] * cos + xshaped[..., 0] * sin,
            ],
            -1,
        )

        return x_out.view_as(x).to(x)

    # def triton_forward(self, x: torch.Tensor, packed_meta_data: PackedMetaData) -> torch.Tensor:
    #     assert False, "Trition RoPE is not implemented yet. Please use the 'eager' implementation for now."
    #     x = apply_rotary_emb(
    #         x,
    #         self._cos_cached,
    #         self._sin_cached,
    #         inplace=True,
    #         interleaved=self.interleaved,
    #         cu_seqlens=packed_meta_data.cu_seq_lens,
    #         max_seqlen=packed_meta_data.max_seq_len,
    #     )
    #     return x
