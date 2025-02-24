import pytest
import torch

from tite.model.pool import PackedAvgPool1d, PackedMetaData


@pytest.mark.parametrize("dim", [1, 4, 8, 16, 64, 768])
@pytest.mark.parametrize("seq_length", [2, 3, 4, 8, 16, 64, 256, 768])
@pytest.mark.parametrize("stride", range(1, 10))
@pytest.mark.parametrize("kernel_size", range(1, 10))
def test_packed_avg_pool1d(
    kernel_size: int, stride: int, seq_length: int, dim: int, dtype: torch.dtype = torch.float32
):
    if stride > kernel_size:
        pytest.skip("Stride must be less than or equal to kernel size")
        return

    eager = PackedAvgPool1d(kernel_size, stride, implementation="eager")
    triton = PackedAvgPool1d(kernel_size, stride, implementation="triton")

    x = torch.randn(2, seq_length, dim, requires_grad=True, device="cuda", dtype=dtype)
    mask = torch.ones(2, seq_length, dtype=torch.bool, device="cuda")
    mask[0, min(-1, -seq_length // 2) :] = False
    seq_lens = mask.sum(-1)
    cu_seq_lens = torch.zeros(mask.shape[0] + 1, dtype=torch.int64, device="cuda")
    cu_seq_lens[1:] = seq_lens.cumsum(0)

    idcs = mask.nonzero(as_tuple=True)
    eager_x = x[idcs].detach().clone().requires_grad_(True)
    triton_x = x[idcs].detach().clone().requires_grad_(True)

    eager_meta_data = PackedMetaData(seq_lens, cu_seq_lens, int(seq_lens.max().int()), idcs)
    triton_meta_data = PackedMetaData(seq_lens, cu_seq_lens, int(seq_lens.max().int()), None)

    output_eager, _ = eager(eager_x, eager_meta_data)
    output_triton, _ = triton(triton_x, triton_meta_data)

    (output_eager * (torch.arange(1, dim + 1, device=x.device, dtype=x.dtype) / dim)).sum().backward()
    (output_triton * (torch.arange(1, dim + 1, device=x.device, dtype=x.dtype) / dim)).sum().backward()

    atol = 1e-6 if dtype == torch.float32 else 1e-2
    assert torch.allclose(output_eager, output_triton, atol=atol)
    assert eager_x.grad is not None and triton_x.grad is not None
    assert torch.allclose(eager_x.grad, triton_x.grad, atol=atol)
