from dataclasses import dataclass
from typing import Literal, Tuple, overload

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
    if stride > kernel_size:
        raise ValueError("Stride must be less than or equal to kernel size")
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
        y = torch.zeros(y_packed_meta_data.cu_seq_lens[-1], dim, device=x.device, dtype=x.dtype)
        norm = torch.zeros_like(y)

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


class PackedAvgPool1d(torch.nn.Module):

    def __init__(self, kernel_size: int, stride: int, implementation: Literal["eager", "triton"] = "triton"):
        super().__init__()
        if stride > kernel_size:
            raise ValueError("Stride must be less than or equal to kernel size")
        self.kernel_size = kernel_size
        self.stride = stride
        self.implementation = implementation

    def eager_forward(self, packed_x: torch.Tensor, packed_meta_data: PackedMetaData) -> torch.Tensor:
        x = torch.zeros(
            len(packed_meta_data.seq_lens),
            packed_meta_data.max_seq_len,
            packed_x.shape[-1],
            device=packed_x.device,
            dtype=packed_x.dtype,
        )
        mask = torch.zeros(
            len(packed_meta_data.seq_lens), packed_meta_data.max_seq_len, device=packed_x.device, dtype=torch.bool
        )
        x[packed_meta_data.idcs] = packed_x
        mask[packed_meta_data.idcs] = True

        x = x.masked_fill(~mask[..., None].expand_as(x), 0)
        batch_size, _, dim = x.shape

        seq_lens = mask.sum(-1)
        if self.kernel_size > x.shape[-2]:
            padding = self.kernel_size - x.shape[-2]
        else:
            padding = (self.kernel_size - seq_lens - self.stride) % self.stride

        output_seq_lens = compute_output_shape(seq_lens, self.kernel_size, self.stride)

        pad_seq_lens = seq_lens + padding
        pad_x = torch.zeros(batch_size, int(pad_seq_lens.max().item()), dim, device=x.device, dtype=x.dtype)
        pad_mask = torch.zeros(batch_size, int(pad_seq_lens.max().item()), device=mask.device, dtype=mask.dtype)
        pad_x[:, : x.shape[-2]] = x
        pad_mask[:, : x.shape[-2]] = mask

        x_blocks = pad_x.unfold(-2, self.kernel_size, self.stride)
        mask_blocks = pad_mask.unfold(-1, self.kernel_size, self.stride).unsqueeze(-2)
        norm = mask_blocks.sum(-1).clamp_min(1)
        y = (x_blocks.sum(-1) / norm).to(packed_x)
        y_mask = torch.arange(y.shape[1], device=y.device)[None].expand(y.shape[0], -1) < output_seq_lens[:, None]
        y_packed = y[y_mask]
        return y_packed

    def forward(self, x: torch.Tensor, packed_meta_data: PackedMetaData) -> Tuple[torch.Tensor, PackedMetaData]:
        if packed_meta_data.max_seq_len == 1 or (self.kernel_size == 1 and self.stride == 1):
            return x, packed_meta_data

        y_seq_lens = compute_output_shape(packed_meta_data.seq_lens, self.kernel_size, self.stride)
        y_max_seq_len = compute_output_shape(packed_meta_data.max_seq_len, self.kernel_size, self.stride)
        y_cu_seq_lens = torch.zeros(y_seq_lens.shape[0] + 1, dtype=packed_meta_data.cu_seq_lens.dtype, device=x.device)
        y_cu_seq_lens[1:] = torch.cumsum(y_seq_lens, dim=0, dtype=packed_meta_data.cu_seq_lens.dtype)
        idcs = packed_meta_data.idcs
        if idcs is not None:
            batch_idcs = torch.arange(len(y_seq_lens), device=x.device).repeat_interleave(y_seq_lens)
            position_idcs = torch.ones(y_cu_seq_lens[-1], device=x.device, dtype=torch.int32)
            position_idcs[y_cu_seq_lens[1:-1]] = -y_seq_lens[:-1] + 1
            position_idcs = position_idcs.cumsum(0) - 1
            idcs = (batch_idcs, position_idcs)

        if self.implementation == "triton":
            y_packed_meta_data = PackedMetaData(y_seq_lens, y_cu_seq_lens, y_max_seq_len, idcs)
            y = ApplyPooling_.apply(x, packed_meta_data, y_packed_meta_data, self.kernel_size, self.stride)
        elif self.implementation == "eager":
            y = self.eager_forward(x, packed_meta_data)
            y_packed_meta_data = PackedMetaData(y_seq_lens, y_cu_seq_lens, y_max_seq_len, idcs)
        return y, y_packed_meta_data
