import pytest
import torch
from transformers import BertConfig, BertModel

from tite.legacy import TiteConfig as TiteLegacyConfig
from tite.legacy import TiteModel as TiteLegacyModel
from tite.model import TiteConfig, TiteModel
from tite.pool import compute_output_shape


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
        positional_embedding_type="absolute",
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


def test_same_as_legacy():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = TiteConfig(
        vocab_size=32,
        num_hidden_layers=6,
        hidden_sizes=(4,) * 6,
        num_attention_heads=(2,) * 6,
        intermediate_sizes=(8,) * 6,
        kernel_sizes=(2,) * 6,
        strides=(2,) * 6,
        max_position_embeddings=64,
        positional_embedding_type="rotary",
        rope_implementation="eager",
        rotary_interleaved=True,
        norm_location="post",
        norm_type="layer",
        attn_implementation="sdpa",
        pooling_implementation="eager",
    )
    model = TiteModel(config).to(device).eval()
    legacy_config = TiteLegacyConfig(
        vocab_size=32,
        num_hidden_layers=6,
        hidden_sizes=(4,) * 6,
        num_attention_heads=(2,) * 6,
        intermediate_sizes=(8,) * 6,
        kernel_sizes=(2,) * 6,
        strides=(2,) * 6,
        max_position_embeddings=64,
        positional_embedding_type="rotary",
    )
    legacy_model = TiteLegacyModel(legacy_config).to(device).eval()

    model.load_state_dict(model._update_state_dict(legacy_model.state_dict()))

    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    attention_mask[0, -config.max_position_embeddings // 2 :] = False

    with torch.amp.autocast(device_type="cuda"):
        output = model(input_ids, attention_mask, output_hidden_states=True)
        legacy_output = legacy_model(input_ids, attention_mask, output_hidden_states=True)
        output.last_hidden_state.sum().backward()
        legacy_output.last_hidden_state.sum().backward()

    seq_lens = attention_mask.sum(-1)
    for i in range(config.num_hidden_layers):
        mask = torch.arange(0, legacy_output.hidden_states[i].shape[1], device=device) < seq_lens[:, None]
        assert torch.allclose(output.hidden_states[i], legacy_output.hidden_states[i][mask], atol=1e-6)
        seq_lens = compute_output_shape(seq_lens, config.kernel_sizes[i], config.strides[i])
    assert torch.allclose(output.last_hidden_state, legacy_output.last_hidden_state, atol=1e-6)
    legacy_grads = {name: param.grad for name, param in legacy_model.named_parameters() if param.grad is not None}
    legacy_grads = model._update_state_dict(legacy_grads)
    for name, param in model.named_parameters():
        assert torch.allclose(legacy_grads[name], param.grad, atol=1e-5)


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
