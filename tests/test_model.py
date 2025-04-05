import pytest
import torch
from transformers import BertConfig, BertModel

from tite.model.pool import compute_output_shape
from tite.model.tite import TiteConfig, TiteForPreTraining, TiteModel


@pytest.fixture
def config() -> TiteConfig:
    config = TiteConfig(
        vocab_size=32,
        num_hidden_layers=3,
        hidden_sizes=(4, 8, 12),
        num_attention_heads=(2, 2, 2),
        intermediate_sizes=(8, 12, 16),
        kernel_sizes=(8, 8, None),
        strides=(2, 1, None),
        max_position_embeddings=16,
        rotary_interleaved=True,
        positional_embedding_type="rotary",
        attn_implementation="eager",
        rope_implementation="eager",
        pooling_implementation="eager",
        compile=False,
    )
    return config


@pytest.mark.parametrize("positional_embedding_type", ["absolute", "alibi", "rotary"])
def test_tite_model(config: TiteConfig, positional_embedding_type: str):
    pytest.MonkeyPatch().setattr(config, "positional_embedding_type", positional_embedding_type)
    model = TiteModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[1, -config.max_position_embeddings // 2 :] = False
    output = model(input_ids, attention_mask).last_hidden_state
    assert output.shape == (2, 1, config.hidden_size)
    assert output.requires_grad


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager", "flash_attention_2"])
@torch.no_grad()
def test_same_as_bert(attn_implementation: str):
    if not torch.cuda.is_available() and attn_implementation == "flash_attention_2":
        pytest.skip("FlashAttention2 is only available on CUDA")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    hidden_size = 16
    num_hidden_layers = 12
    num_attention_heads = 2
    intermediate_size = 64
    bert = (
        BertModel(
            BertConfig(
                hidden_size=hidden_size,
                num_hidden_layer=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attn_implementation=attn_implementation if attn_implementation != "flash_attention_2" else "sdpa",
            )
        )
        .eval()
        .to(device)
    )
    bert.embeddings.token_type_embeddings.weight.zero_()
    tite = (
        TiteModel(
            TiteConfig(
                hidden_sizes=(hidden_size,) * num_hidden_layers,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=(num_attention_heads,) * num_hidden_layers,
                intermediate_sizes=(intermediate_size,) * num_hidden_layers,
                kernel_sizes=(None,) * num_hidden_layers,
                strides=(None,) * num_hidden_layers,
                hidden_act="gelu",
                positional_embedding_type="absolute",
                norm_location="post",
                norm_type="layer",
                attn_implementation=attn_implementation,
                compile=False,
            )
        )
        .eval()
        .to(device)
    )
    config = tite.config

    tite.load_state_dict(tite._update_state_dict(bert.state_dict()))

    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    attention_mask[1, -config.max_position_embeddings // 2 :] = False

    with torch.autocast(device_type="cuda"):
        bert_output = bert(
            input_ids, attention_mask, output_hidden_states=True, output_attentions=attn_implementation == "eager"
        )
        tite_output = tite(
            input_ids, attention_mask, output_hidden_states=True, output_attentions=attn_implementation == "eager"
        )

    for i in range(config.num_hidden_layers):
        if tite_output.attentions is not None:
            assert torch.allclose(
                tite_output.attentions[i][attention_mask[:, None, :, None].expand_as(bert_output.attentions[i])],
                bert_output.attentions[i][attention_mask[:, None, :, None].expand_as(bert_output.attentions[i])],
                atol=1e-4,
            )
        assert torch.allclose(tite_output.hidden_states[i], bert_output.hidden_states[i][attention_mask], atol=1e-4)
    assert torch.allclose(
        tite_output.last_hidden_state[attention_mask], bert_output.last_hidden_state[attention_mask], atol=1e-4
    )


def test_pretrain(config: TiteConfig):
    model = TiteForPreTraining(config)

    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[1, -config.max_position_embeddings // 2 :] = False
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    special_tokens_mask[1, -config.max_position_embeddings // 2 :] = True

    labels = {}
    for task, head in model.heads.items():
        labels[task] = head.get_labels(input_ids, attention_mask, special_tokens_mask)

    output = model(input_ids, attention_mask, labels=labels)

    assert output.losses is not None
