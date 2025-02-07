import pytest
import torch
from transformers import BertConfig, BertModel

from tite.model import TiteConfig, TiteModel


@pytest.fixture
def config() -> TiteConfig:
    config = TiteConfig(
        vocab_size=32,
        num_hidden_layers=3,
        hidden_size=(4, 6, 8),
        num_attention_heads=(2, 2, 2),
        intermediate_size=(8, 12, 16),
        kernel_size=(8, 8, None),
        stride=(2, 1, None),
        max_position_embeddings=16,
        positional_embedding_type="absolute",
    )
    return config


@pytest.mark.parametrize("positional_embedding_type", ["absolute", "ALiBi"])
def test_tite_model(config: TiteConfig, positional_embedding_type: str):
    pytest.MonkeyPatch().setattr(config, "positional_embedding_type", positional_embedding_type)
    model = TiteModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[1, -config.max_position_embeddings // 2 :] = False
    output = model(input_ids, attention_mask).last_hidden_state
    assert output.shape == (2, 1, config.last_hidden_size)
    assert output.requires_grad


@torch.no_grad()
def test_same_as_bert():
    hidden_size = 16
    num_hidden_layers = 2
    num_attention_heads = 2
    intermediate_size = 64
    bert = BertModel(
        BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        )
    ).eval()
    bert.embeddings.token_type_embeddings.weight.zero_()
    tite = TiteModel(
        TiteConfig(
            hidden_sizes=(hidden_size,) * num_hidden_layers,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=(num_attention_heads,) * num_hidden_layers,
            intermediate_sizes=(intermediate_size,) * num_hidden_layers,
            kernel_sizes=(None,) * num_hidden_layers,
            strides=(None,) * num_hidden_layers,
            positional_embedding_type="absolute",
            pre_norm=False,
            norm_type="layer",
            attn_implementation="sdpa",
        )
    ).eval()
    config = tite.config

    tite.load_state_dict(tite._update_state_dict(bert.state_dict()))

    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))

    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[1, -config.max_position_embeddings // 2 :] = False

    bert_output = bert(input_ids, attention_mask, output_hidden_states=True)
    tite_output = tite(input_ids, attention_mask, output_hidden_states=True)

    for i in range(config.num_hidden_layers):
        assert torch.allclose(tite_output.hidden_states[i], bert_output.hidden_states[i][attention_mask], atol=1e-6)
