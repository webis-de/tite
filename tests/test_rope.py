import torch

from tite.pool import PackedMetaData
from tite.rope import EagerRotaryPositionalEmbeddings, TritonRotaryPositionalEmbeddings


def test_rope():
    batch_size = 2
    seq_len = 128
    num_attention_heads = 12
    head_dim = 64
    hidden_states = torch.rand(batch_size * seq_len, num_attention_heads, head_dim, device="cuda", dtype=torch.float16)
    eager_rope = EagerRotaryPositionalEmbeddings(head_dim, 512, dtype=hidden_states.dtype).to("cuda")
    rope = TritonRotaryPositionalEmbeddings(head_dim).to("cuda")

    seq_lens = torch.tensor([seq_len] * batch_size, device="cuda", dtype=torch.int32)
    cu_seq_lens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    cu_seq_lens[1:] = seq_lens.cumsum(0)
    packed_meta_data = PackedMetaData(seq_lens, cu_seq_lens, seq_len, None)

    eager_rope_hidden_states = eager_rope(hidden_states, packed_meta_data)
    rope_hidden_states = rope(hidden_states, packed_meta_data)

    assert torch.allclose(eager_rope_hidden_states, rope_hidden_states, atol=1e-6)
