import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import normalize
from transformers import PreTrainedTokenizerBase

from .bert import BertModel
from .datasets import GLUEDataModule
from .glue_module import GlueModule
from .jepa import JEPA, LossFn, Predictor
from .predictor import MLMDecoder
from .transformations import Transformation


class _DetachFromGrad(Module):
    def __init__(self, module: Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._module = module

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            output = self._module(*args, **kwargs)
        assert isinstance(output, Tensor)
        return output.detach()  # Better safe than sorry


class TiteModule(LightningModule):
    def __init__(
        self,
        student: Module,
        teacher: Module | None,
        tokenizer: PreTrainedTokenizerBase,
        predictor: Predictor,
        loss: LossFn,
        text_key: str = "text",
        max_length: int | None = None,
        detach_teacher_from_grad: bool = False,
        student_transformations: list[Transformation] | None = None,
        teacher_transformations: list[Transformation] | Literal["student"] | None = None,
    ) -> None:
        super().__init__()
        # ties weights for BERT models -- only works for teacher MLM and student BERT
        if isinstance(student, BertModel) and isinstance(predictor, MLMDecoder):
            student.tie_decoder_weights(predictor.decoder)
        if teacher is None:
            teacher = student
        self._student = student
        self._teacher = teacher
        self._tokenizer = tokenizer
        self._student_transforms = student_transformations or []
        self._teacher_transforms = (
            self._student_transforms if teacher_transformations == "student" else (teacher_transformations or [])
        )
        self._predictor = predictor
        self._loss = loss
        self._text_key = text_key
        self._max_length = max_length
        if detach_teacher_from_grad:
            self._teacher = _DetachFromGrad(self._teacher)
        self._jepa = JEPA(self._student, self._teacher, predictor, loss, return_embeddings=True)

        self._tokens_seen = 0.0

    def on_validation_start(self) -> None:
        if self.trainer is not None and self.trainer.limit_val_batches == 0:
            return
        # Train on GLUE
        # for task in TASK_COLUMN_NAMES:
        for task in ["mrpc"]:
            glue = GLUEDataModule(task=task, batch_size=32, tokenizer=self._tokenizer)
            glue_module = GlueModule(deepcopy(self._student).train(), self._tokenizer, glue.hparams.name)
            trainer = Trainer(
                logger=False,
                precision=(self.trainer.precision if self.trainer is not None else "bf16-mixed"),
                max_epochs=10,
                # callbacks=[EarlyStopping(glue_module._evaluation_metrics[0].__class__.__name__, mode="max", patience=1)],
                enable_checkpointing=False,
            )
            trainer.fit(glue_module, glue)
            metrics = trainer.logged_metrics
            for name, value in metrics.items():
                self.log(f"{glue.hparams.name}/{name}", value, on_step=False, on_epoch=True)

    def validation_step(self, batch: dict[str, Any] | None) -> None:
        # Empty validation step to trick pytorch lightning into validating this model though validation is actually done
        # using the GlueModule
        return

    def training_step(self, batch: dict[str, torch.Tensor]) -> Tensor:
        student_input = batch
        teacher_input = batch
        aux = {}
        for transformation in self._student_transforms:
            transformed = transformation(**student_input)
            student_input = transformed[0]
            aux = {**aux, **transformed[1]}
        for transformation in self._teacher_transforms:
            transformed = transformation(**teacher_input)
            teacher_input = transformed[0]
        # JEPA will try to predict the original from the transformed input within the embedding space, i.e.,
        #   Loss(pred(student(studentinput), aux), teacher(teacherinput))
        jepa_loss, embs, embt = self._jepa(student_input, teacher_input, **aux)
        attention_mask = batch["attention_mask"]
        num_tokens = attention_mask.sum() if attention_mask is not None else batch["input_ids"].numel()
        self._tokens_seen += num_tokens
        self.log("tokens_seen", self._tokens_seen, on_step=True, reduce_fx="max")  # We sum it up ourselves
        self.log_dict({"loss": jepa_loss}, prog_bar=True, on_step=True)
        ####
        # Log additional metrics for more insight into the training
        with torch.autocast(device_type="cuda", enabled=False):
            cossim = normalize(embs[:, 0]) @ normalize(embt[:, 0]).T
            crossentropy = torch.nn.functional.cross_entropy(
                (embs[:, 0] @ embt[:, 0].T) / math.sqrt(embs.shape[-1]), torch.arange(embs.shape[0], device=self.device)
            )
            # Equivalent to above: -torch.diag(torch.log_softmax(embs[:, 0] @ embt[:, 0].T, dim=, -1)).mean()
            crosscorr = normalize(embs[:, 0], dim=0).T @ normalize(embt[:, 0], dim=0) / embt.shape[0]
        metrics = {"crossentropy": crossentropy, "pairwise-cossim": cossim, "crosscorrelation": crosscorr}
        for metric_name, metric_value in metrics.items():
            if metric_value.ndim > 1:
                if self.logger is not None and (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
                    self.logger.log_image(metric_name, [metric_value])
            else:
                self.log(metric_name, metric_value)
        ####
        return jepa_loss

    def save_pretrained(self, save_path: str | Path) -> None:
        self._student.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)

    def on_save_checkpoint(self, *args, **kwargs) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.save_pretrained(save_path)
