import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities import grad_norm
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerBase

from .bert import BertModel
from .datasets import GLUEDataModule, IRDatasetsDataModule
from .glue_module import GlueModule
from .jepa import JEPA, LossFn, Predictor
from .msmarco_module import MSMARCOModule
from .predictor import MLMDecoder


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
        detach_teacher_from_grad: bool = False,
        log_additional_metrics: bool = False,
        validate_on_glue: bool = False,
        validate_on_msmarco: bool = False,
        log_gradients: bool = False,
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
        self._log_additional_metrics = log_additional_metrics
        self._validate_on_glue = validate_on_glue
        self._validate_on_msmarco = validate_on_msmarco
        self._log_gradients = log_gradients

        self._predictor = predictor
        self._loss = loss
        if detach_teacher_from_grad:
            self._teacher = _DetachFromGrad(self._teacher)
        self._jepa = JEPA(self._student, self._teacher, predictor, loss, return_embeddings=True)

        self._tokens_seen = 0.0

    def on_validation_start(self) -> None:
        if self.trainer is None:
            return
        if self.trainer.limit_val_batches == 0:
            return
        add_special_tokens = self.trainer.datamodule.collator._add_special_tokens
        # Train on GLUE
        if self._validate_on_glue:
            # for task in TASK_COLUMN_NAMES:
            for task in ["mrpc"]:
                glue = GLUEDataModule(
                    task=task, tokenizer=self._tokenizer, batch_size=32, add_special_tokens=add_special_tokens
                )
                glue_module = GlueModule(deepcopy(self._student).train(), self._tokenizer, glue.hparams.name)
                trainer = Trainer(
                    logger=False,
                    precision=(self.trainer.precision if self.trainer is not None else "bf16-mixed"),
                    max_epochs=10,
                    # callbacks=[EarlyStopping(glue_module._evaluation_metrics[0].__class__.__name__, mode="max", patience=1)],
                    enable_checkpointing=False,
                    num_sanity_val_steps=0,
                )
                trainer.fit(glue_module, glue)
                metrics = trainer.logged_metrics
                for name, value in metrics.items():
                    self.log(f"{glue.hparams.name}/{name}", value, on_step=False, on_epoch=True)
        if self._validate_on_msmarco:
            msmarco = IRDatasetsDataModule(
                "msmarco-passage",
                tokenizer=self._tokenizer,
                add_special_tokens=add_special_tokens,
                trainset=("train/triples-small", "triples"),
                valset=("trec-dl-2019/judged", "scoreddocs"),
                batch_size=32,
            )
            msmarco_module = MSMARCOModule(deepcopy(self._student).train(), self._tokenizer)
            max_steps = 5_000
            trainer = Trainer(
                logger=False,
                precision=(self.trainer.precision if self.trainer is not None else "bf16-mixed"),
                max_steps=max_steps,
                # callbacks=[EarlyStopping("MeanSquaredError", mode="min", patience=1)],
                enable_checkpointing=False,
                num_sanity_val_steps=0,
                val_check_interval=max_steps,
            )
            trainer.fit(msmarco_module, msmarco)
            metrics = trainer.logged_metrics
            for name, value in metrics.items():
                self.log(f"trec-dl-2019/{name}", value, on_step=False, on_epoch=True)

    def on_before_optimizer_step(self, optimizer):
        if not self._log_gradients:
            return
        norms = grad_norm(self._student, norm_type=2)
        self.log_dict(norms)

    def validation_step(self, batch: dict[str, Any] | None) -> None:
        # Empty validation step to trick pytorch lightning into validating this model though validation is actually done
        # using the GlueModule
        return

    def training_step(self, batch: dict[str, torch.Tensor]) -> Tensor:
        student_input = batch.pop("student_input")
        teacher_input = batch.pop("teacher_input", None)
        # JEPA will try to predict the original from the transformed input within the embedding space, i.e.,
        #   Loss(pred(student(studentinput), aux), teacher(teacherinput))
        jepa_loss, embs, embt = self._jepa(student_input, teacher_input, **batch)
        num_tokens = max(
            student_input["attention_mask"].sum().item(),
            0 if teacher_input is None else teacher_input["attention_mask"].sum().item(),
        )
        self._tokens_seen += num_tokens
        self.log("tokens_seen", self._tokens_seen, on_step=True, reduce_fx="max")  # We sum it up ourselves
        self.log_dict({"loss": jepa_loss}, prog_bar=True, on_step=True)
        ####
        # Log additional metrics for more insight into the training
        if self._log_additional_metrics:
            with torch.autocast(device_type="cuda", enabled=False):
                # cossim = normalize(embs[:, 0]) @ normalize(embt[:, 0]).T
                crossentropy = torch.nn.functional.cross_entropy(
                    (embs[:, 0] @ embt[:, 0].T) / math.sqrt(embs.shape[-1]),
                    torch.arange(embs.shape[0], device=self.device),
                )
                # Equivalent to above: -torch.diag(torch.log_softmax(embs[:, 0] @ embt[:, 0].T, dim=, -1)).mean()
                # crosscorr = normalize(embs[:, 0], dim=0).T @ normalize(embt[:, 0], dim=0) / embt.shape[0]
            metrics = {
                "crossentropy": crossentropy,
                # "pairwise-cossim": cossim,
                # "crosscorrelation": crosscorr,
            }
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
