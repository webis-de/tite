from pathlib import Path
from typing import Literal

import pytest

from tite.datasets.fineweb import TransformationCollator
from tite.decoder import BOWDecoder, Decoder, MAEEnhancedDecoder, MLMDecoder
from tite.loss import BOWBinaryCrossEntropyLoss, LossFunction, MAECrossEntropyLoss, MLMCrossEntropyLoss
from tite.model.bert import BertConfig, BertModel
from tite.model.tite import TiteConfig, TiteModel
from tite.model.tokenizer import TiteTokenizer
from tite.module import TiteModule
from tite.teacher import BOWTeacher, MAEEnhancedTeacher, MLMTeacher, Teacher
from tite.transformation import StringTransformation, TokenMLMMask, TokenTransformation

DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    (
        "config,"
        "decoders,"
        "teachers,"
        "encoder_string_transformations,"
        "decoder_string_transformations,"
        "encoder_token_transformations,"
        "decoder_token_transformations,"
        "loss_functions"
    ),
    [
        (
            BertConfig(
                vocab_size=32,
                num_hidden_layers=2,
                hidden_size=4,
                num_attention_heads=2,
                intermediate_size=8,
                max_position_embeddings=16,
                max_length=16,
                positional_embedding_type="absolute",
                attn_implementation="sdpa",
                pooling_implementation="eager",
                rope_implementation="eager",
            ),
            [MLMDecoder(vocab_size=32, hidden_size=4)],
            [MLMTeacher(pad_id=0)],
            None,
            [None],
            [TokenMLMMask(vocab_size=32, mask_id=0, mask_prob=0.3)],
            [None],
            [MLMCrossEntropyLoss(vocab_size=32)],
        ),
        (
            TiteConfig(
                vocab_size=32,
                num_hidden_layers=2,
                hidden_sizes=(4, 4),
                num_attention_heads=(2, 2),
                intermediate_sizes=(8, 8),
                kernel_sizes=(8, 8),
                strides=(2, 1),
                max_position_embeddings=16,
                max_length=16,
                positional_embedding_type="absolute",
                attn_implementation="sdpa",
                pooling_implementation="eager",
                rope_implementation="eager",
            ),
            [
                MAEEnhancedDecoder(
                    vocab_size=32, hidden_size=4, num_attention_heads=2, intermediate_size=8, mask_prob=0.5
                ),
                BOWDecoder(vocab_size=32, hidden_size=4),
            ],
            [MAEEnhancedTeacher(pad_id=0), BOWTeacher(vocab_size=32, pad_id=0)],
            None,
            [None, None],
            None,  # no masking necessary -- masking is handled by the maeenhaced decoder
            [None, None],
            [MAECrossEntropyLoss(vocab_size=32), BOWBinaryCrossEntropyLoss(vocab_size=32)],
        ),
    ],
    ids=["bert", "tite"],
)
def test_training_step(
    config: TiteConfig,
    decoders: list[Decoder],
    teachers: list[Teacher],
    encoder_string_transformations: list[StringTransformation] | None,
    decoder_string_transformations: list[list[StringTransformation] | Literal["encoder"] | None],
    encoder_token_transformations: list[TokenTransformation] | None,
    decoder_token_transformations: list[list[TokenTransformation] | Literal["encoder"] | None],
    loss_functions: list[LossFunction],
):
    if isinstance(config, BertConfig):
        encoder = BertModel(config)
    else:
        encoder = TiteModel(config)
    tokenizer_dir = DATA_DIR / "tokenizer"
    tokenizer = TiteTokenizer(
        str(tokenizer_dir / "vocab.txt"),
        str(tokenizer_dir / "tokenizer.json"),
        model_max_length=config.max_length,
    )
    module = TiteModule(encoder, tokenizer, decoders, teachers, loss_functions)
    collator = TransformationCollator(
        tokenizer,
        ("text", None),
        encoder_string_transformations,
        decoder_string_transformations,
        encoder_token_transformations,
        decoder_token_transformations,
    )

    batch = collator([{"text": "hello world"}, {"text": "goodbye world"}])

    loss = module.training_step(batch=batch)
    assert loss is not None
    assert loss.requires_grad
    assert loss > 0
