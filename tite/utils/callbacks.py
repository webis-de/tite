from lightning import Callback
from lightning_ir.modeling_utils.mlm_head import (
    MODEL_TYPE_TO_HEAD_NAME,
    MODEL_TYPE_TO_LM_HEAD,
    MODEL_TYPE_TO_OUTPUT_EMBEDDINGS,
    MODEL_TYPE_TO_TIED_WEIGHTS_KEYS,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer

from tite.model.legacy import TiteConfig as TiteLegacyConfig
from tite.model.legacy import TiteModel as TiteLegacyModel
from tite.model.tite import BOWLMHead, TiteConfig, TiteModel
from tite.model.tokenizer import TiteTokenizer

AutoConfig.register(TiteConfig.model_type, TiteConfig)
AutoModel.register(TiteConfig, TiteModel)
AutoTokenizer.register(TiteConfig, None, TiteTokenizer)

AutoConfig.register(TiteLegacyConfig.model_type, TiteLegacyConfig)
AutoModel.register(TiteLegacyConfig, TiteLegacyModel)
AutoTokenizer.register(TiteLegacyConfig, None, TiteTokenizer)


MODEL_TYPE_TO_LM_HEAD["tite"] = BOWLMHead

MODEL_TYPE_TO_HEAD_NAME["tite"] = "heads.bow_auto_encoding"

MODEL_TYPE_TO_OUTPUT_EMBEDDINGS["tite"] = "projection.lm_head.decoder"

MODEL_TYPE_TO_TIED_WEIGHTS_KEYS["tite"] = ["projection.lm_head.decoder.bias", "projection.lm_head.decoder.weight"]


class DummyImportCallback(Callback):
    pass
