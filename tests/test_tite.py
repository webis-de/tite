from pathlib import Path

import pytest
import torch

from tite.model.tite import TiteConfig, TiteForPreTraining, TiteModel


# @pytest.mark.parametrize("positional_embedding_type", ["absolute", "alibi", "rotary"])
@pytest.mark.parametrize("pooling_implementation", ["eager", "triton"])
@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_tite_model(
    config: TiteConfig,
    # positional_embedding_type: str,
    pooling_implementation: str,
    attn_implementation: str,
):
    # pytest.MonkeyPatch().setattr(config, "positional_embedding_type", positional_embedding_type)
    pytest.MonkeyPatch().setattr(config, "pooling_implementation", pooling_implementation)
    pytest.MonkeyPatch().setattr(config, "_attn_implementation", attn_implementation)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = TiteModel(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    attention_mask[0, -config.max_position_embeddings // 2 :] = False
    output = model(input_ids, attention_mask).last_hidden_state
    assert output.shape == (2, 1, config.hidden_size)
    assert output.requires_grad


@pytest.mark.parametrize("pooling_location", ["pre", "intra", "post"])
@pytest.mark.parametrize("norm_location", ["pre", "post"])
def test_pooling_and_norm_locations(config: TiteConfig, pooling_location: str, norm_location: str):
    pytest.MonkeyPatch().setattr(config, "pooling_location", pooling_location)
    pytest.MonkeyPatch().setattr(config, "norm_location", norm_location)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = TiteModel(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    attention_mask[0, -config.max_position_embeddings // 2 :] = False
    output = model(input_ids, attention_mask).last_hidden_state
    assert output.shape == (2, 1, config.hidden_size)
    assert output.requires_grad


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

    output = model(input_ids, attention_mask, original_input_ids=input_ids, labels=labels)

    assert output.losses is not None


def test_upscale(config: TiteConfig, tmp_path: Path):

    assert config.hidden_size > config.hidden_sizes[0]
    model_for_pretraining = TiteForPreTraining(config).eval()

    model_for_pretraining.save_pretrained(tmp_path / "model")

    model = TiteModel.from_pretrained(tmp_path / "model", attn_implementation="sdpa").eval()

    input_ids = torch.randint(0, config.vocab_size, (2, config.max_position_embeddings))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[1, -config.max_position_embeddings // 2 :] = False
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    special_tokens_mask[1, -config.max_position_embeddings // 2 :] = True

    output_for_pretraining = model_for_pretraining(input_ids, attention_mask)
    output = model(input_ids, attention_mask)

    assert (output_for_pretraining.last_hidden_state == output.last_hidden_state).all()
