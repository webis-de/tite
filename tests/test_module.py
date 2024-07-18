from pathlib import Path

import pytest
from torch.nn import Module

from tite.bert import BertConfig, BertModel
from tite.loss import BarlowTwins, MLMCrossEntropy
from tite.model import TiteConfig, TiteModel
from tite.module import TiteModule
from tite.predictor import Identity, MLMDecoder
from tite.teacher import MLMPredictor
from tite.tokenizer import TiteTokenizer
from tite.transformations import MaskTokens, Transformation

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def config() -> TiteConfig:
    return TiteConfig(
        vocab_size=32,
        num_hidden_layers=2,
        hidden_size=(4, 4),
        num_attention_heads=(2, 2),
        intermediate_size=(8, 8),
        kernel_size=(None, None),
        stride=(None, None),
        max_position_embeddings=16,
    )


@pytest.mark.parametrize(
    "config,teacher,predictor,transformation,loss",
    [
        (
            BertConfig(
                vocab_size=32,
                num_hidden_layers=2,
                hidden_size=4,
                num_attention_heads=2,
                intermediate_size=8,
                max_position_embeddings=16,
            ),
            MLMPredictor(0),
            MLMDecoder(32, 4),
            MaskTokens(4, 2, 3),
            MLMCrossEntropy(32),
        ),
        (
            TiteConfig(
                vocab_size=32,
                num_hidden_layers=2,
                hidden_size=(4, 4),
                num_attention_heads=(2, 2),
                intermediate_size=(8, 8),
                kernel_size=(8, 8),
                stride=(2, 1),
                max_position_embeddings=16,
            ),
            None,
            Identity(),
            MaskTokens(4, 2, 3),
            BarlowTwins(0.1, 4),
        ),
    ],
    ids=["bert", "tite"],
)
def test_training_step(
    config: TiteConfig,
    teacher: Module | None,
    predictor: Module,
    transformation: Transformation,
    loss: Module,
):
    if isinstance(config, BertConfig):
        student = BertModel(config, predictor)
    else:
        student = TiteModel(config)
    tokenizer_dir = DATA_DIR / "tokenizer"
    tokenizer = TiteTokenizer(
        str(tokenizer_dir / "vocab.txt"),
        str(tokenizer_dir / "tokenizer.json"),
    )
    module = TiteModule(student, teacher, tokenizer, [transformation], predictor, loss)

    loss = module.training_step({"text": ["1 2 3 4", "1 2 3 4 5"]})
    assert loss is not None
    assert loss.requires_grad
    assert loss > 0
    loss.backward()
    for name, param in list(student.named_parameters())[::-1]:
        assert param.grad is not None
        assert not param.grad.isnan().any()
