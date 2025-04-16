import pytest
import torch

from tite.model.bert import BertConfig, BertForPreTraining, BertModel


@pytest.fixture
def config() -> BertConfig:
    config = BertConfig(
        vocab_size=32,
        num_hidden_layers=3,
        hidden_size=4,
        num_attention_heads=2,
        intermediate_size=8,
        max_position_embeddings=16,
        rotary_interleaved=True,
        positional_embedding_type="rotary",
        attn_implementation="eager",
        rope_implementation="eager",
        pooling_implementation="eager",
    )
    return config


def test_pretrain(config: BertConfig):
    model = BertForPreTraining(config)

    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[1, -config.max_position_embeddings // 2 :] = False
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    special_tokens_mask[1, -config.max_position_embeddings // 2 :] = True
    mlm_mask = torch.rand(input_ids.shape) < 0.3

    labels = {}
    for task, head in model.heads.items():
        labels[task] = head.get_labels(input_ids, attention_mask, special_tokens_mask, mlm_mask=mlm_mask)

    output = model(input_ids, attention_mask, original_input_ids=input_ids, labels=labels)

    assert output.losses is not None
