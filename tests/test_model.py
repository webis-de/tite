from copy import deepcopy

import pytest
import torch
from transformers import BertConfig, BertModel

from tite.model import MaskedAvgPool1d, TiteConfig, TiteModel, compute_output_shapes


@pytest.fixture
def config() -> TiteConfig:
    config = TiteConfig(
        vocab_size=32,
        num_hidden_layers=2,
        hidden_size=(4, 4),
        num_attention_heads=(2, 2),
        intermediate_size=(8, 8),
        kernel_size=(8, 8),
        stride=(2, 1),
        max_position_embeddings=16,
        positional_embedding_type="absolute",
    )
    return config


@pytest.mark.parametrize("kernel_size, stride, seq_length", [(3, 1, 8), (3, 2, 8), (3, 3, 8)])
def test_masked_avg_pool1d(kernel_size: int, stride: int, seq_length: int):
    layer = MaskedAvgPool1d(kernel_size, stride)

    x = torch.randn(2, seq_length, 4)
    mask = torch.ones(2, seq_length, dtype=torch.bool)
    mask[0, -seq_length // 2 :] = False

    output_shapes = compute_output_shapes(seq_length, (kernel_size,), (stride,))

    output, output_mask = layer(x, mask)
    assert output.shape[1] == output_shapes[-1]
    assert ((output != 0).all(-1) == output_mask).all()
    assert output_mask.shape[1] == output_shapes[-1]


@pytest.mark.parametrize("positional_embedding_type", ["absolute", "ALiBi"])
def test_tite_model(config: TiteConfig, positional_embedding_type: str):
    pytest.MonkeyPatch().setattr(config, "positional_embedding_type", positional_embedding_type)
    model = TiteModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[1, -config.max_position_embeddings // 2 :] = False
    output = model(input_ids, attention_mask)
    assert output.shape == (2, 1, config.last_hidden_size)
    assert output.requires_grad


def test_same_as_bert(config: TiteConfig):
    config = deepcopy(config)
    config.kernel_size = (None, None)
    config.stride = (None, None)
    model = TiteModel(config).eval()

    bert_kwargs = config.to_dict()
    bert_kwargs["hidden_size"] = bert_kwargs["hidden_size"][0]
    bert_kwargs["num_attention_heads"] = bert_kwargs["num_attention_heads"][0]
    bert_kwargs["intermediate_size"] = bert_kwargs["intermediate_size"][0]
    bert_kwargs.pop("kernel_size")
    bert_kwargs.pop("stride")
    bert_config = BertConfig(**bert_kwargs)
    bert_model = BertModel(bert_config).eval()

    bert_model.embeddings.token_type_embeddings.weight.data.zero_()

    missing_keys = []
    qkv_weight = []
    qkv_bias = []
    for key, value in bert_model.state_dict().items():
        if key in model.state_dict():
            model.state_dict()[key].copy_(value)
        else:
            if "query" in key or "key" in key or "value" in key:
                if "weight" in key:
                    qkv_weight.append(value)
                else:
                    qkv_bias.append(value)
            else:
                missing_keys.append(key)
            if "value" in key:
                if "weight" in key:
                    model.state_dict()[key.replace("value", "Wqkv")].copy_(torch.cat(qkv_weight, 0))
                    qkv_weight = []
                if "bias" in key:
                    model.state_dict()[key.replace("value", "Wqkv")].copy_(torch.cat(qkv_bias, 0))
                    qkv_bias = []
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))

    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[1, -config.max_position_embeddings // 2 :] = False

    with torch.no_grad():
        tite_output = model(input_ids, attention_mask)
        bert_output = bert_model(input_ids, attention_mask)

    assert torch.allclose(tite_output[attention_mask], bert_output.last_hidden_state[attention_mask], atol=1e-6)
