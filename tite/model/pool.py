from dataclasses import dataclass
from typing import Literal, Tuple, overload

import torch

try:
    import triton
    import triton.language as tl

    _has_triton = True
except ImportError:
    _has_triton = False

    class triton:
        def jit(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class tl:
        constexpr = None


from .unpad import pad_input


@overload
def ceil_div(a: int, b: int) -> int: ...
@overload
def ceil_div(a: torch.Tensor, b: int) -> torch.Tensor: ...


def ceil_div(a, b):
    return -(-a // b)


@overload
def compute_output_shape(input_shape: int, kernel_size: int | None, stride: int | None) -> int: ...
@overload
def compute_output_shape(input_shape: torch.Tensor, kernel_size: int | None, stride: int | None) -> torch.Tensor: ...


def compute_output_shape(input_shape, kernel_size, stride):
    if kernel_size is None or stride is None:
        return input_shape
    if stride > kernel_size:
        raise ValueError("Stride must be less than or equal to kernel size")
    if isinstance(input_shape, int):
        return 0 if input_shape == 0 else ceil_div((max(0, input_shape - kernel_size)), stride) + 1
    elif isinstance(input_shape, torch.Tensor):
        return torch.where(input_shape == 0, 0, ceil_div((torch.clamp(input_shape - kernel_size, min=0)), stride) + 1)
    else:
        raise NotImplementedError(f"Unsupported type {type(input_shape)}")


def cat_arange(arange_starts: torch.Tensor, arange_ends: torch.Tensor) -> torch.Tensor:
    arange_lengths = arange_ends - arange_starts
    offsets = torch.cumsum(arange_lengths, dim=0) - arange_lengths - arange_starts
    return torch.arange(arange_lengths.sum(), device=arange_lengths.device) - torch.repeat_interleave(
        offsets, arange_lengths
    )


@dataclass
class PackedMetaData:
    seq_lens: torch.Tensor
    cu_seq_lens: torch.Tensor
    max_seq_len: int
    _idcs: torch.Tensor | None = None
    _position_idcs: torch.Tensor | None = None

    @classmethod
    @torch.compiler.disable
    def from_attention_mask(cls, attention_mask: torch.Tensor) -> "PackedMetaData":
        seq_lens = attention_mask.sum(-1, dtype=torch.int32)
        idcs = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seq_len = attention_mask.shape[-1]
        cu_seq_lens = torch.nn.functional.pad(torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0))
        position_idcs = cat_arange(torch.zeros_like(seq_lens), seq_lens)
        return cls(seq_lens, cu_seq_lens, max_seq_len, _idcs=idcs, _position_idcs=position_idcs)

    @property
    def position_idcs(self):
        if self._position_idcs is None:
            self._position_idcs = cat_arange(torch.zeros_like(self.seq_lens), self.seq_lens)
        return self._position_idcs

    @property
    def idcs(self):
        if self._idcs is None:
            idcs_starts = torch.arange(
                0, self.seq_lens.shape[0] * self.max_seq_len, self.max_seq_len, device=self.seq_lens.device
            )
            idcs_ends = idcs_starts + self.seq_lens
            self._idcs = cat_arange(idcs_starts, idcs_ends)
        return self._idcs


@triton.jit
def forward_pooling_kernel(
    # Pointers to matrices
    X,
    Y,
    N,
    X_CU_SEQ_LENS,
    Y_CU_SEQ_LENS,
    # Matrix dimensions
    dim,
    # strides
    stride_x_seq_len,
    stride_x_dim,
    stride_y_seq_len,
    stride_y_dim,
    # Meta-parameters
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
):
    pid_dim = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    X_start_idx = tl.load(X_CU_SEQ_LENS + pid_batch)
    Y_start_idx = tl.load(Y_CU_SEQ_LENS + pid_batch)
    x_seq_len = tl.load(X_CU_SEQ_LENS + pid_batch + 1) - X_start_idx
    y_seq_len = tl.load(Y_CU_SEQ_LENS + pid_batch + 1) - Y_start_idx
    X = X + X_start_idx * stride_x_seq_len
    Y = Y + Y_start_idx * stride_y_seq_len
    N = N + Y_start_idx * stride_y_seq_len

    if pid_seq * BLOCK_S >= y_seq_len:
        return
    idx_y_seq = pid_seq * BLOCK_S + tl.arange(0, BLOCK_S)
    idx_dim = pid_dim * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    norm = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    for i in range(kernel_size):
        idx_x_seq = idx_y_seq[:, None] * stride + i
        X_ = X + (idx_x_seq * stride_x_seq_len + idx_dim[None, :] * stride_x_dim)
        mask = (idx_x_seq < x_seq_len) & (idx_dim[None, :] < dim)
        x = tl.load(X_, mask=mask, other=0.0)
        acc += x
        norm += mask.to(tl.float32)

    norm = 1 / norm
    acc *= norm

    Y = Y + idx_y_seq[:, None] * stride_y_seq_len + idx_dim[None, :] * stride_y_dim
    N = N + idx_y_seq[:, None] * stride_y_seq_len + idx_dim[None, :] * stride_y_dim
    tl.store(Y, acc, mask=(idx_y_seq[:, None] < y_seq_len) & (idx_dim[None, :] < dim))
    tl.store(N, norm, mask=(idx_y_seq[:, None] < y_seq_len) & (idx_dim[None, :] < dim))


def apply_forward_pooling(
    x: torch.Tensor,
    y: torch.Tensor,
    norm: torch.Tensor,
    x_cu_seq_lens: torch.Tensor,
    y_cu_seq_lens: torch.Tensor,
    y_max_seq_len: int,
    kernel_size: int,
    stride: int,
) -> None:
    batch = x_cu_seq_lens.shape[0] - 1
    dim = x.shape[-1]

    BLOCK_D = min(triton.next_power_of_2(dim), 1024)
    BLOCK_S = min(triton.next_power_of_2(y_max_seq_len), 4)
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_D"]), triton.cdiv(y_max_seq_len, META["BLOCK_S"]), batch)  # noqa

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        forward_pooling_kernel[grid](
            x,  # data ptrs
            y,
            norm,
            x_cu_seq_lens,
            y_cu_seq_lens,
            dim,  # shapes
            x.stride(0),  # strides
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_D,  # constants
            BLOCK_S,
            kernel_size,
            stride,
        )


@triton.jit
def div_fl(a, b):
    # division with floor
    # a // b does not work for negative numbers in Triton
    # (it rounds upwards for negative numbers)
    return (a - (b - 1)) // b


@triton.jit
def backward_pooling_kernel(
    # Pointers to matrices
    X,
    Y,
    N,
    X_CU_SEQ_LENS,
    Y_CU_SEQ_LENS,
    # Matrix dimensions
    dim,
    # strides
    stride_x_seq_len,
    stride_x_dim,
    stride_y_seq_len,
    stride_y_dim,
    # Meta-parameters
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    n_padding: tl.constexpr,
):
    # the backward kernel indexes the gradient tensor as a dilated tensor
    # see the following article for details
    # https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
    pid_dim = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    X_start_idx = tl.load(X_CU_SEQ_LENS + pid_batch)
    Y_start_idx = tl.load(Y_CU_SEQ_LENS + pid_batch)
    x_seq_len = tl.load(X_CU_SEQ_LENS + pid_batch + 1) - X_start_idx
    y_seq_len = tl.load(Y_CU_SEQ_LENS + pid_batch + 1) - Y_start_idx
    X = X + X_start_idx * stride_x_seq_len
    N = N + X_start_idx * stride_x_seq_len
    Y = Y + Y_start_idx * stride_y_seq_len

    if pid_seq * BLOCK_S >= y_seq_len:
        return
    idx_y_seq = pid_seq * BLOCK_S + tl.arange(0, BLOCK_S)
    rk = pid_dim * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    for i in range(kernel_size):
        idx_x_seq = idx_y_seq[:, None] + i - padding
        dilation_mask = (idx_x_seq % stride) == 0
        padding_mask = (idx_x_seq >= 0) & (idx_x_seq < x_seq_len + (x_seq_len - 1) * dilation)
        idx_x_seq = div_fl(idx_x_seq, stride)
        X_ = X + (idx_x_seq * stride_x_seq_len + rk[None, :] * stride_x_dim)
        x_mask = padding_mask & dilation_mask & (rk[None, :] < dim)

        idx_n_seq = div_fl(idx_y_seq[:, None] + i - n_padding, stride)
        N_ = N + (idx_n_seq * stride_x_seq_len + rk[None, :] * stride_x_dim)
        n_mask = (idx_n_seq >= 0) & (idx_n_seq < x_seq_len) & (rk[None, :] < dim)

        x = tl.load(X_, mask=x_mask, other=0.0)
        n = tl.load(N_, mask=n_mask, other=0.0)
        acc += x * n

    Y = Y + idx_y_seq[:, None] * stride_y_seq_len + rk[None, :] * stride_y_dim
    tl.store(Y, acc, mask=(idx_y_seq[:, None] < y_seq_len) & (rk[None, :] < dim))


def apply_backward_pooling(
    x: torch.Tensor,
    y: torch.Tensor,
    norm: torch.Tensor,
    x_cu_seq_lens: torch.Tensor,
    y_cu_seq_lens: torch.Tensor,
    y_max_seq_len: int,
    kernel_size: int,
    stride: int,
) -> None:
    batch = x_cu_seq_lens.shape[0] - 1
    dim = x.shape[-1]

    BLOCK_D = min(triton.next_power_of_2(dim), 1024)
    BLOCK_S = min(triton.next_power_of_2(y_max_seq_len), 4)
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_D"]), triton.cdiv(y_max_seq_len, META["BLOCK_S"]), batch)  # noqa

    padding = kernel_size - 1
    dilation = stride - 1
    # center the kernel for the normalization of the backward pass
    if kernel_size == stride:
        n_padding = 0
    else:
        n_padding = max(1, kernel_size - 2 * stride + 1)

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        backward_pooling_kernel[grid](
            x,  # data ptrs
            y,
            norm,
            x_cu_seq_lens,
            y_cu_seq_lens,
            dim,  # shapes
            x.stride(0),  # strides
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_D,  # constants
            BLOCK_S,
            kernel_size,
            stride,
            padding,
            dilation,
            n_padding,
        )


class ApplyPooling_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        x_packed_meta_data: PackedMetaData,
        y_packed_meta_data: PackedMetaData,
        kernel_size: int,
        stride: int,
    ) -> torch.Tensor:

        dim = x.shape[-1]
        y = torch.empty(y_packed_meta_data.cu_seq_lens[-1], dim, device=x.device, dtype=x.dtype)
        norm = torch.empty_like(y)

        apply_forward_pooling(
            x=x,
            y=y,
            norm=norm,
            x_cu_seq_lens=x_packed_meta_data.cu_seq_lens,
            y_cu_seq_lens=y_packed_meta_data.cu_seq_lens,
            y_max_seq_len=y_packed_meta_data.max_seq_len,
            kernel_size=kernel_size,
            stride=stride,
        )
        ctx.save_for_backward(x, x_packed_meta_data.cu_seq_lens, y_packed_meta_data.cu_seq_lens, norm)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.x_max_seq_len = x_packed_meta_data.max_seq_len
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, x_cu_seq_lens, y_cu_seq_lens, norm = ctx.saved_tensors
        x_max_seq_len = ctx.x_max_seq_len
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        grad_x = torch.zeros_like(x)
        apply_backward_pooling(
            x=grad_y,
            y=grad_x,
            norm=norm,
            x_cu_seq_lens=y_cu_seq_lens,
            y_cu_seq_lens=x_cu_seq_lens,
            y_max_seq_len=x_max_seq_len,
            kernel_size=kernel_size,
            stride=stride,
        )
        return grad_x, None, None, None, None


class MaskedAvgPool1d(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        if stride > kernel_size:
            raise ValueError("Stride must be less than or equal to kernel size")
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.kernel_size > x.shape[-2]:
            padding = self.kernel_size - x.shape[-2]
        else:
            padding = (self.kernel_size - x.shape[-2] - self.stride) % self.stride

        if padding != 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            mask = torch.nn.functional.pad(mask, (0, padding))
        x_blocks = x.unfold(-2, self.kernel_size, self.stride)
        mask_blocks = mask.unfold(-1, self.kernel_size, self.stride).unsqueeze(-2)
        x_masked = x_blocks * mask_blocks
        normalization = mask_blocks.sum(-1)
        normalization[normalization == 0] = 1
        y = x_masked.sum(-1) / normalization
        output_seq_lens = compute_output_shape(mask.sum(-1), self.kernel_size, self.stride)
        y_mask = torch.arange(y.shape[1], device=y.device)[None].expand(y.shape[0], -1) < output_seq_lens[:, None]
        return y, y_mask


class PackedAvgPool1d(MaskedAvgPool1d):

    def __init__(self, kernel_size: int, stride: int, implementation: Literal["eager", "triton"] = "triton"):
        super().__init__(kernel_size, stride)
        if stride > kernel_size:
            raise ValueError("Stride must be less than or equal to kernel size")
        if implementation == "triton" and not _has_triton:
            raise ValueError("Triton is not installed. Please install it to use the 'triton' implementation.")
        self.implementation = implementation

    def eager_forward(self, packed_x: torch.Tensor, packed_meta_data: PackedMetaData) -> torch.Tensor:
        x = pad_input(packed_x, packed_meta_data)
        mask = pad_input(torch.ones(packed_x.shape[0], dtype=torch.bool, device=packed_x.device), packed_meta_data)

        y, y_mask = super().forward(x, mask)

        y_packed = y[y_mask]
        return y_packed

    def triton_forward(self, packed_x: torch.Tensor, packed_meta_data: PackedMetaData) -> torch.Tensor:
        y = ApplyPooling_.apply(packed_x, packed_meta_data, packed_meta_data, self.kernel_size, self.stride)
        return y

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, packed_meta_data: PackedMetaData) -> Tuple[torch.Tensor, PackedMetaData]:
        if packed_meta_data.max_seq_len == 1 or (self.kernel_size == 1 and self.stride == 1):
            return x, packed_meta_data

        y_seq_lens = compute_output_shape(packed_meta_data.seq_lens, self.kernel_size, self.stride)
        y_max_seq_len = compute_output_shape(packed_meta_data.max_seq_len, self.kernel_size, self.stride)
        y_cu_seq_lens = torch.nn.functional.pad(torch.cumsum(y_seq_lens, dim=0, dtype=torch.int32), (1, 0))

        y_packed_meta_data = PackedMetaData(y_seq_lens, y_cu_seq_lens, y_max_seq_len)
        if self.implementation == "triton":
            y = ApplyPooling_.apply(x, packed_meta_data, y_packed_meta_data, self.kernel_size, self.stride)
        elif self.implementation == "eager":
            y = self.eager_forward(x, packed_meta_data)
        return y, y_packed_meta_data
