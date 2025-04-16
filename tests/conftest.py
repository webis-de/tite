from pathlib import Path

import pytest

from tite.model.tite import TiteConfig
from tite.model.tokenizer import TiteTokenizer

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def tokenizer() -> TiteTokenizer:
    tokenizer = TiteTokenizer.from_pretrained(DATA_DIR / "tokenizer")
    return tokenizer


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
