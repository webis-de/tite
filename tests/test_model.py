import pytest
import torch
from transformers import BertConfig, BertModel
from copy import deepcopy

from tite.model.model import (
    MaskedAvgPool1d,
    TiteConfig,
    TiteModel,
    compute_output_shape,
)


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
    )
    return config


@pytest.mark.parametrize(
    "kernel_size, stride, seq_length", [(3, 1, 8), (3, 2, 8), (3, 3, 8)]
)
def test_masked_avg_pool1d(kernel_size: int, stride: int, seq_length: int):
    layer = MaskedAvgPool1d(kernel_size, stride)

    x = torch.randn(2, seq_length, 4)
    mask = torch.ones(2, seq_length)
    mask[0, -seq_length // 2 :] = 0

    output_shape = compute_output_shape(seq_length, (kernel_size,), (stride,))

    output, output_mask = layer(x, mask)
    assert output.shape[1] == output_shape
    assert ((output != 0).all(-1) == output_mask).all()
    assert output_mask.shape[1] == output_shape


def test_tite_model(config: TiteConfig):
    model = TiteModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))
    output = model(input_ids)
    assert output.shape == (2, 1, config.hidden_size[-1])
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
    for key, value in bert_model.state_dict().items():
        if key in model.state_dict():
            model.state_dict()[key].copy_(value)
        else:
            missing_keys.append(key)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))

    with torch.no_grad():
        tite_output = model(input_ids)
        bert_output = bert_model(input_ids)

    assert torch.allclose(tite_output, bert_output.last_hidden_state, atol=1e-4)
