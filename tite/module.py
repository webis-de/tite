from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerBase, TensorType

from .jepa import JEPA, LossFn, Predictor
from .model import TiteModel
from .transformations import Transformation


class _DetachFromGrad(Module):

    def __init__(self, module: Module, *args, **kwargs) -> None:
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
        teacher: Module | None,
        tokenizer: PreTrainedTokenizerBase,
        transformations: list[Transformation],
        predictor: Predictor,
        loss: LossFn,
        text_key: str = "text",
    ) -> None:
        super().__init__()
        if teacher is None:
            teacher = student
        self._student = student
        self._teacher = teacher
        self._tokenizer = tokenizer
        self._transforms = transformations
        self._predictor = predictor
        self._loss = loss
        self._text_key = text_key
        self._jepa = JEPA(
            self._student, _DetachFromGrad(self._teacher), predictor, loss
        )

    def training_step(self, batch: dict[str, Any]):
        tokenized = self._tokenizer(
            text=batch[self._text_key],
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            return_tensors=TensorType.PYTORCH,
            truncation=True,
        ).to(self.device)
        for transformation in self._transforms:
            transformed = transformation(**tokenized)[0]
        jepa_loss = self._jepa(tokenized, transformed, None)
        self.log_dict({"loss": jepa_loss}, prog_bar=True, on_step=True)
        return jepa_loss

    def save_pretrained(self, save_path: str | Path) -> None:
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def on_save_checkpoint(self, *args, **kwargs) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            _step = self.trainer.global_step
            self.config.save_step = _step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.save_pretrained(save_path)

    # def configure_optimizers(self) -> Optimizer:
    #     return AdamW(self._student.parameters())
