from dataclasses import dataclass
from typing import Tuple, overload

import torch
import triton
import triton.language as tl


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
    if isinstance(input_shape, int):
        return ceil_div((max(0, input_shape - kernel_size)), stride) + 1
    elif isinstance(input_shape, torch.Tensor):
        return ceil_div((torch.clamp(input_shape - kernel_size, min=0)), stride) + 1
    else:
        raise NotImplementedError(f"Unsupported type {type(input_shape)}")


@dataclass
class PackedMetaData:
    seq_lens: torch.Tensor
    cu_seq_lens: torch.Tensor
    max_seq_len: int
    idcs: Tuple[torch.Tensor, ...] | None


@triton.jit
def forward_pooling_kernel(
    # Pointers to matrices
    X,
    Y,
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
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    X_start_idx = tl.load(X_CU_SEQ_LENS + pid_batch)
    Y_start_idx = tl.load(Y_CU_SEQ_LENS + pid_batch)
    x_seq_len = tl.load(X_CU_SEQ_LENS + pid_batch + 1) - X_start_idx
    y_seq_len = tl.load(Y_CU_SEQ_LENS + pid_batch + 1) - Y_start_idx
    X = X + X_start_idx * stride_x_seq_len
    Y = Y + Y_start_idx * stride_y_seq_len

    if pid_m * BLOCK_M >= y_seq_len:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for i in range(kernel_size):
        rm_ = rm[:, None] * stride + i
        X_ = X + (rm_ * stride_x_seq_len + rk[None, :] * stride_x_dim)
        x = tl.load(X_, mask=(rm_ < x_seq_len) & (rk[None, :] < dim), other=0.0)
        acc += x

    acc /= kernel_size

    Y = Y + rm[:, None] * stride_y_seq_len + rk[None, :] * stride_y_dim
    tl.store(Y, acc, mask=(rm[:, None] < y_seq_len) & (rk[None, :] < dim))


def apply_forward_pooling(
    x: torch.Tensor,
    y: torch.Tensor,
    x_cu_seq_lens: torch.Tensor,
    y_cu_seq_lens: torch.Tensor,
    y_max_seq_len: int,
    kernel_size: int,
    stride: int,
) -> None:
    batch = x_cu_seq_lens.shape[0] - 1
    dim = x.shape[-1]

    BLOCK_M = 8 if dim <= 128 else 4
    BLOCK_K = 32 if dim <= 32 else (64 if dim <= 64 else (128 if dim <= 128 else 256))
    grid = lambda META: (triton.cdiv(y_max_seq_len, META["BLOCK_M"]), batch, triton.cdiv(dim, META["BLOCK_K"]))  # noqa

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        forward_pooling_kernel[grid](
            x,  # data ptrs
            y,
            x_cu_seq_lens,
            y_cu_seq_lens,
            dim,  # shapes
            x.stride(0),  # strides
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_K,  # constants
            BLOCK_M,
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
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
):
    # the backward kernel indexes the gradient tensor as a dilated tensor
    # see the following article for details
    # https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    X_start_idx = tl.load(X_CU_SEQ_LENS + pid_batch)
    Y_start_idx = tl.load(Y_CU_SEQ_LENS + pid_batch)
    x_seq_len = tl.load(X_CU_SEQ_LENS + pid_batch + 1) - X_start_idx
    y_seq_len = tl.load(Y_CU_SEQ_LENS + pid_batch + 1) - Y_start_idx
    X = X + X_start_idx * stride_x_seq_len
    Y = Y + Y_start_idx * stride_y_seq_len

    padding = kernel_size - 1
    dilation = stride - 1

    if pid_m * BLOCK_M >= y_seq_len:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for i in range(kernel_size):
        rm_ = rm[:, None] + i - padding
        dilation_mask = (rm_ % stride) == 0
        padding_mask = (rm_ >= 0) & (rm_ < x_seq_len + (x_seq_len - 1) * dilation)
        rm_ = div_fl(rm_, stride)
        X_ = X + (rm_ * stride_x_seq_len + rk[None, :] * stride_x_dim)
        x = tl.load(X_, mask=padding_mask & dilation_mask & (rk[None, :] < dim), other=0.0)
        acc += x

    acc /= kernel_size

    Y = Y + rm[:, None] * stride_y_seq_len + rk[None, :] * stride_y_dim
    tl.store(Y, acc, mask=(rm[:, None] < y_seq_len) & (rk[None, :] < dim))


def apply_backward_pooling(
    x: torch.Tensor,
    y: torch.Tensor,
    x_cu_seq_lens: torch.Tensor,
    y_cu_seq_lens: torch.Tensor,
    y_max_seq_len: int,
    kernel_size: int,
    stride: int,
) -> None:
    batch = x_cu_seq_lens.shape[0] - 1
    dim = x.shape[-1]

    BLOCK_M = 8 if dim <= 128 else 4
    BLOCK_K = 32 if dim <= 32 else (64 if dim <= 64 else (128 if dim <= 128 else 256))
    grid = lambda META: (triton.cdiv(y_max_seq_len, META["BLOCK_M"]), batch, triton.cdiv(dim, META["BLOCK_K"]))  # noqa

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        backward_pooling_kernel[grid](
            x,  # data ptrs
            y,
            x_cu_seq_lens,
            y_cu_seq_lens,
            dim,  # shapes
            x.stride(0),  # strides
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_K,  # constants
            BLOCK_M,
            kernel_size,
            stride,
        )


class ApplyPooling_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, packed_meta_data: PackedMetaData, kernel_size: int, stride: int
    ) -> Tuple[torch.Tensor, PackedMetaData]:
        y_seq_lens = compute_output_shape(packed_meta_data.seq_lens, kernel_size, stride)
        y_max_seq_len = compute_output_shape(packed_meta_data.max_seq_len, kernel_size, stride)
        y_cu_seq_lens = torch.zeros(y_seq_lens.shape[0] + 1, dtype=packed_meta_data.cu_seq_lens.dtype, device=x.device)
        y_cu_seq_lens[1:] = torch.cumsum(y_seq_lens, dim=0, dtype=packed_meta_data.cu_seq_lens.dtype)
        y_packed_meta_data = PackedMetaData(y_seq_lens, y_cu_seq_lens, y_max_seq_len)

        dim = x.shape[-1]
        y = torch.zeros(y_cu_seq_lens[-1], dim, device=x.device, dtype=x.dtype)

        apply_forward_pooling(
            x=x,
            y=y,
            x_cu_seq_lens=packed_meta_data.cu_seq_lens,
            y_cu_seq_lens=y_cu_seq_lens,
            y_max_seq_len=y_max_seq_len,
            kernel_size=kernel_size,
            stride=stride,
        )
        ctx.save_for_backward(x, packed_meta_data.cu_seq_lens, y_packed_meta_data.cu_seq_lens)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.x_max_seq_len = packed_meta_data.max_seq_len
        return y, y_packed_meta_data

    @staticmethod
    def backward(ctx, grad_y, grad_meta_data):
        x, x_cu_seq_lens, y_cu_seq_lens = ctx.saved_tensors
        x_max_seq_len = ctx.x_max_seq_len
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        grad_x = torch.zeros_like(x)
        apply_backward_pooling(
            x=grad_y,
            y=grad_x,
            x_cu_seq_lens=y_cu_seq_lens,
            y_cu_seq_lens=x_cu_seq_lens,
            y_max_seq_len=x_max_seq_len,
            kernel_size=kernel_size,
            stride=stride,
        )
        return grad_x, None, None, None


class PackedAvgPool1d(torch.nn.Module):

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor, packed_meta_data: PackedMetaData) -> Tuple[torch.Tensor, PackedMetaData]:
        if packed_meta_data.max_seq_len == 1:
            return x, packed_meta_data
        return ApplyPooling_.apply(x, packed_meta_data, self.kernel_size, self.stride)
