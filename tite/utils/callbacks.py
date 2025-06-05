from lightning import Callback
from transformers import AutoConfig, AutoModel, AutoTokenizer

from tite.model.tite import TiteConfig, TiteModel
from tite.model.tokenizer import TiteTokenizer

AutoConfig.register(TiteConfig.model_type, TiteConfig)
AutoModel.register(TiteConfig, TiteModel)
AutoTokenizer.register(TiteConfig, None, TiteTokenizer)


class DummyImportCallback(Callback):
    pass
