from typing import Any, Callable, Iterable

from lightning import LightningModule
from torch import LongTensor
from transformers import PreTrainedTokenizerBase

from .model.model import TiteModel

Transformation = Callable[[tuple[LongTensor, LongTensor]], Any]


class TiteModule(LightningModule):

    def __init__(
        self,
        model: TiteModel,
        tokenizer: PreTrainedTokenizerBase,
        transformation: list[Transformation],
        text_key: str = "text",
    ):
        super().__init__()
        self.model = model

    def foreard(self, batch: Iterable[dict[str, Any]]):
        raise NotImplementedError
        texts, meta = zip(
            *((d[self._text_key], [d[k] for k in self._meta_keys]) for d in batch)
        )
        tokenized = self._tokenizer(
            text=texts,
            return_attention_mask=True,
            padding=True,
            return_tensors=TensorType.PYTORCH,
        )
        metadata = default_collate(meta)
