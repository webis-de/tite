import torch

from tite.decoder.decoder import MAEEnhancedDecoder
from tite.model.legacy import MAEEnhancedDecoder as LegacyMAEEnhancedDecoder
from tite.model.tite import TiteModel


def test_mae_same_as_legacy():
    batch_size = 2
    seq_len = 16
    hidden_size = 4
    num_attention_heads = 1
    intermediate_size = 8
    mask_prob = 0
    legacy_decoder = LegacyMAEEnhancedDecoder(
        hidden_size, num_attention_heads, intermediate_size, mask_id=0, mask_prob=mask_prob
    ).eval()
    decoder = MAEEnhancedDecoder(
        hidden_size, num_attention_heads, intermediate_size, mask_id=0, mask_prob=mask_prob
    ).eval()
    state_dict = TiteModel._update_state_dict(legacy_decoder.state_dict())
    decoder.load_state_dict(state_dict)

    input_ids = torch.randint(1, 32, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    embx = torch.rand(batch_size, 1, hidden_size)

    legacy_output = legacy_decoder(embx, input_ids, attention_mask)
    output = decoder(embx, input_ids, attention_mask)

    assert torch.allclose(legacy_output, output)
