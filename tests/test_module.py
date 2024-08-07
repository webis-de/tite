from pathlib import Path

import pytest
import torch
from torch.nn import Module
from transformers import AutoTokenizer, BertForMaskedLM

from tite.bert import BertConfig, BertModel
from tite.loss import BarlowTwins, MLMCrossEntropy
from tite.loss.mlm import MLMCrossEntropy
from tite.model import TiteConfig, TiteModel
from tite.module import TiteModule
from tite.predictor import Identity, MLMDecoder
from tite.teacher import MLMTeacher
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
                max_length=16,
            ),
            MLMTeacher(0),
            MLMDecoder(32, 4),
            MaskTokens(32, 4, 2, 3),
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
                max_length=16,
            ),
            None,
            Identity(),
            MaskTokens(32, 4, 2, 3),
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
        student = BertModel(config)
    else:
        student = TiteModel(config)
    tokenizer_dir = DATA_DIR / "tokenizer"
    tokenizer = TiteTokenizer(
        str(tokenizer_dir / "vocab.txt"),
        str(tokenizer_dir / "tokenizer.json"),
        model_max_length=config.max_length,
    )
    module = TiteModule(student, teacher, tokenizer, [transformation], predictor, loss)

    loss = module.training_step(
        {
            "text": [
                "5 6 7 8 9 10 11 12 13 15 16 17 18 19 20",
            ]
            * 8
            + ["5 6 7 8"]
        }
    )
    assert loss is not None
    assert loss.requires_grad
    assert loss > 0
    loss.backward()
    for name, param in list(student.named_parameters())[::-1]:
        assert param.grad is not None
        assert not param.grad.isnan().any()


class DeterministicMaskTokens(MaskTokens):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.LongTensor, **kwargs) -> list[dict]:
        torch.manual_seed(0)
        return super().forward(input_ids, attention_mask, **kwargs)


def test_same_as_bert_loss():
    texts = [
        (
            "Discover the cosmos! Each day a different image or photograph of our fascinating universe is featured, "
            "along with a brief explanation written by a professional astronomer.\n 2007 February 11\nExplanation: "
            "What's happening on Jupiter's moon Io? Two sulfurous eruptions are visible on Jupiter's volcanic moon "
            "Io in this color composite image from the robotic Galileo spacecraft that orbited Jupiter from 1995 to "
            "2003. At the image top, over Io's limb, a bluish plume rises about 140 kilometers above the surface of a "
            "volcanic caldera known as Pillan Patera. In the image middle, near the night/day shadow line, the ring "
            "shaped Prometheus plume is seen rising about 75 kilometers above Io while casting a shadow below the "
            "volcanic vent. Named for the Greek god who gave mortals fire, the Prometheus plume is visible in every "
            "image ever made of the region dating back to the Voyager flybys of 1979 - presenting the possibility that "
            "this plume has been continuously active for at least 18 years. The above digitally sharpened image was "
            "originally recorded in 1997 on June 28 from a distance of about 600,000 kilometers.\nAuthors & editors:\n"
            "Jerry Bonnell (USRA)\nNASA Official: Phillip Newman Specific rights apply.\nA service of: ASD at NASA / "
            "GSFC\n& Michigan Tech. U."
        ),
        (
            "Congestion pricing is a type of tolling created to manage traffic congestion.\nFor almost 50 years, "
            "economists have been advocating Congestion Pricing (where toll prices rise and fall based on the number "
            "of cars on the road) as the most effective way to balance supply and demand on highways.\nThey argue the "
            "economic and social costs of congestion are far greater than costs associated with tolling.\nTypical "
            "driver behavior (where many drivers enter highways at the same time, a.k.a “rush hour”) assumes all "
            "drivers have equal values of time.\nHowever, this has been widely disproved. People are different, and "
            "they have different needs when it comes to driving.\nCongestion Pricing works to accommodate these needs "
            "with varying toll prices.\nIt's currently used on select highly congested highways in some states, "
            "including:\nDownload the brochure\nPDF, 399 KB)\n- New York\nIt's important to note congestion pricing is "
            "not about collecting money. It's about getting commuters to shift the time they make discretionary "
            "(work-related) trips, so severe traffic congestion can be reduced or eliminated.\nSeveral congestion "
            "management strategies are used in Virginia, including conventional toll roads and open road tolling.\nThe "
            "495 Express Lanes on Interstate 495 (the Capital Beltway) in Northern Virginia are an example of "
            "congestion pricing.\nDownload the fact sheet\n(PDF, 399 KB)\n- Tolling, Congestion Priced Tolling, and "
            "Electronic Tolling in Hampton Roads, Virginia Summary: May 2008 (58 KB)\n- About congestion pricing "
            "(80 KB)\n- Benefits (85 KB)\n- Frequently asked questions (65 KB)\n- Congestion pricing in the U.S. "
            "(73 KB)\n- Resources (69 KB)\n- Electronic tolling (74 KB)\n- Congestion management strategies in "
            "Virginia (65 KB)\n- Hampton Roads Electronic Tolling/E-ZPass Study (1 MB)"
        ),
    ]

    orig_model = BertForMaskedLM.from_pretrained("bert-base-uncased").eval()
    orig_model.bert.embeddings.token_type_embeddings.weight.data.zero_()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    orig_config = orig_model.config
    # num_hidden_layers = orig_config.num_hidden_layers
    config = TiteConfig(positional_embedding_type="absolute", hidden_act=orig_config.hidden_act)
    model = TiteModel(config).eval()
    state_dict = {}
    qkv_weight = []
    qkv_bias = []
    for key, value in orig_model.state_dict().items():
        if not key.startswith("bert.") or "token_type_embeddings" in key:
            continue
        key = key[5:]
        if "query" in key or "key" in key or "value" in key:
            if "weight" in key:
                qkv_weight.append(value)
            else:
                qkv_bias.append(value)
        else:
            state_dict[key] = value
        if "value" in key:
            if "weight" in key:
                state_dict[key.replace("value", "Wqkv")] = torch.cat(qkv_weight, 0)
                qkv_weight = []
            if "bias" in key:
                state_dict[key.replace("value", "Wqkv")] = torch.cat(qkv_bias, 0)
                qkv_bias = []
    model.load_state_dict(state_dict)
    teacher = MLMTeacher(config.pad_token_id)
    transformation = DeterministicMaskTokens(
        len(tokenizer), tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, 0.3
    )
    predictor = MLMDecoder(config.vocab_size, config.hidden_size[0], config.hidden_act)
    state_dict = {}
    for key, value in orig_model.cls.state_dict().items():
        state_dict[key.replace("predictions.", "").replace("transform.", "")] = value
    predictor.load_state_dict(state_dict)

    loss = MLMCrossEntropy(config.vocab_size)
    module = TiteModule(model, teacher, tokenizer, [transformation], predictor, loss)

    encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    transformed, aux = transformation(encoding["input_ids"], encoding["attention_mask"])
    labels = teacher(encoding["input_ids"], aux[0]["mlm_mask"])
    encoding["input_ids"] = transformed[0]["input_ids"]
    orig_loss = orig_model(**encoding, labels=labels).loss
    loss = module.training_step({"text": texts})

    assert torch.allclose(loss, orig_loss, atol=1e-6)
