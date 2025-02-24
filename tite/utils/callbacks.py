from lightning import Callback
from transformers import AutoConfig, AutoModel, AutoTokenizer

from tite.model.legacy import TiteConfig as TiteLegacyConfig
from tite.model.legacy import TiteModel as TiteLegacyModel
from tite.model.tite import TiteConfig, TiteModel
from tite.model.tokenizer import TiteTokenizer

AutoConfig.register(TiteConfig.model_type, TiteConfig)
AutoModel.register(TiteConfig, TiteModel)
AutoTokenizer.register(TiteConfig, None, TiteTokenizer)

AutoConfig.register(TiteLegacyConfig.model_type, TiteLegacyConfig)
AutoModel.register(TiteLegacyConfig, TiteLegacyModel)
AutoTokenizer.register(TiteLegacyConfig, None, TiteTokenizer)


class DummyImportCallback(Callback):
    pass
