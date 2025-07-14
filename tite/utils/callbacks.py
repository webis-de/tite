from lightning import Callback
from lightning_ir.bi_encoder.bi_encoder_tokenizer import ADD_MARKER_TOKEN_MAPPING
from transformers import AutoConfig, AutoModel, AutoTokenizer

from tite.model.tite import TiteConfig, TiteModel
from tite.model.tokenizer import TiteTokenizer

AutoConfig.register(TiteConfig.model_type, TiteConfig)
AutoModel.register(TiteConfig, TiteModel)
AutoTokenizer.register(TiteConfig, None, TiteTokenizer)


ADD_MARKER_TOKEN_MAPPING["tite"] = {
    "single": "{TOKEN} $0",
    "pair": "{TOKEN_1} $A [SEP] {TOKEN_2} $B:1",
}


class DummyImportCallback(Callback):
    pass
