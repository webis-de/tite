from typing import Any, Callable

import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from transformers import PreTrainedTokenizerBase, TensorType

from .jepa import JEPA
from .loss import BarlowTwins
from .model.model import TiteModel


def _tite_jepa_predictor(x, aux):
    return x


class _DetachFromGrad(nn.Module):

    def __init__(self, module: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._module = module

    def forward(self, *args, **kwargs) -> Tensor:
        output = self._module(*args, **kwargs)
        assert isinstance(output, Tensor)
        return output.detach()


class TiteModule(LightningModule):

    def __init__(
        self,
        student: TiteModel,
        tokenizer: PreTrainedTokenizerBase,
        transformation: Callable[[Any], list[dict]],
        text_key: str = "text",
    ) -> None:
        super().__init__()
        self._student = student
        self._tokenizer = tokenizer
        self._transform = transformation
        self._text_key = text_key
        self._jepa = JEPA(
            self._student,
            _DetachFromGrad(self._student),
            _tite_jepa_predictor,
            BarlowTwins(0.5, 768),
        )

    def training_step(self, batch: dict[str, Any]):
        tokenized = self._tokenizer(
            text=batch[self._text_key],
            return_attention_mask=True,
            padding=True,
            return_tensors=TensorType.PYTORCH,
        )
        # TODO: support multiple transformations
        transformed = self._transform(**tokenized)[0]
        jepa_loss = self._jepa(tokenized, transformed, None)
        return jepa_loss

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self._student.parameters())
