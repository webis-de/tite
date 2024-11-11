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

from .datasets import GLUEDataModule, IRDatasetsDataModule
from .glue_module import GlueModule
from .jepa import JEPA, LossFn
from .lars import LARS
from .lr_schedulers import LARSScheduler
from .model import TiteModel
from .msmarco_module import MSMARCOModule
from .predictor import MAEDecoder, MLMDecoder


class _DetachFromGrad(Module):
    def __init__(self, module: Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = module

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            output = self.module(*args, **kwargs)
        assert isinstance(output, Tensor)
        return output.detach()  # Better safe than sorry


class TiteModule(LightningModule):
    def __init__(
        self,
        student: Module,
        teachers: list[Module | None],
        tokenizer: PreTrainedTokenizerBase,
        predictors: list[Module],
        losses: list[LossFn],
        detach_teacher_from_grad: bool = False,
        log_additional_metrics: bool = False,
        validate_on_glue: bool = False,
        validate_on_msmarco: bool = False,
        log_gradients: bool = False,
    ) -> None:
        super().__init__()
        # ties weights for BERT models -- only works for teacher MLM and student BERT
        if (
            len(predictors) == 1
            and isinstance(predictors[0], (MLMDecoder, MAEDecoder))
            and isinstance(student, TiteModel)
        ):
            student.tie_decoder_weights(predictors[0].decoder)
            if isinstance(predictors[0], MAEDecoder):
                predictors[0].embeddings.word_embeddings = student.get_input_embeddings()
                if student.embeddings.position_embeddings is not None:
                    predictors[0].embeddings.position_embeddings = student.embeddings.position_embeddings
        self.student = student
        self.teachers = torch.nn.ModuleList([teacher if teacher is not None else student for teacher in teachers])
        self.tokenizer = tokenizer
        self.log_additional_metrics = log_additional_metrics
        self.validate_on_glue = validate_on_glue
        self.validate_on_msmarco = validate_on_msmarco
        self.log_gradients = log_gradients

        self.predictors = torch.nn.ModuleList(predictors)
        self.losses = torch.nn.ModuleList(losses)
        if detach_teacher_from_grad:
            self.teacher = _DetachFromGrad(self.teacher)
        self.jepa = JEPA(self.student, self.teachers, self.predictors, self.losses)

        self.tokens_seen = 0.0

    def on_validation_start(self) -> None:
        if self.trainer is None:
            return
        if self.trainer.limit_val_batches == 0:
            return
        add_special_tokens = self.trainer.datamodule.collator.add_special_tokens
        # Train on GLUE
        if self.validate_on_glue:
            # for task in TASK_COLUMN_NAMES:
            for task in ["mrpc"]:
                glue = GLUEDataModule(
                    task=task,
                    tokenizer=self.tokenizer,
                    batch_size=32,
                    add_special_tokens=add_special_tokens,
                    streaming=False,
                )
                copy_student = deepcopy(self.student).train()
                if hasattr(copy_student.config, "pooling") and getattr(copy_student.config, "pooling") is None:
                    copy_student.config.pooling = "first"
                glue_module = GlueModule(copy_student, self.tokenizer, glue.hparams.name)
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
                    if "step" in name:
                        continue
                    self.log(f"{glue.hparams.name}/{name}", value, on_step=False, on_epoch=True)
        if self.validate_on_msmarco:
            msmarco = IRDatasetsDataModule(
                tokenizer=self.tokenizer,
                add_special_tokens=add_special_tokens,
                trainset=("msmarco-passage/train/triples-small", "triples"),
                valset=("msmarco-passage/trec-dl-2019/judged", "scoreddocs"),
                batch_size=32,
                inference_batch_size=256,
            )
            copy_student = deepcopy(self.student).train()
            if hasattr(copy_student.config, "pooling") and getattr(copy_student.config, "pooling") is None:
                copy_student.config.pooling = "first"
            msmarco_module = MSMARCOModule(copy_student, self.tokenizer)
            max_steps = 1_000
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
                if "step" in name:
                    continue
                self.log(f"trec-dl-2019/{name}", value, on_step=False, on_epoch=True)

    def on_before_optimizer_step(self, optimizer):
        for name, module in (
            [("student", self.student)]
            + [(f"teacher_{idx}", teacher) for idx, teacher in enumerate(self.teachers)]
            + [(f"predictor_{idx}", predictor) for idx, predictor in enumerate(self.predictors)]
        ):
            norms = grad_norm(module, norm_type=2)
            if not norms:
                continue
            total_norm = norms["grad_2.0_norm_total"]
            module_norms = {f"{name}_grad_2.0_norm_total": total_norm}
            if self.log_gradients:
                self.log_dict(module_norms)

    def validation_step(self, batch: dict[str, Any] | None) -> None:
        # Empty validation step to trick pytorch lightning into validating this model though validation is actually done
        # using the GlueModule
        return

    def training_step(self, batch: dict[str, torch.Tensor]) -> Tensor:
        student_input = batch.pop("student_input")
        teacher_input = batch.pop("teacher_input", None)
        # JEPA will try to predict the original from the transformed input within the embedding space, i.e.,
        #   Loss(pred(student(studentinput), aux), teacher(teacherinput))
        losses, embs = self.jepa(student_input, teacher_input, **batch)
        losses["total"] = sum(losses.values())
        num_tokens = max(
            student_input["attention_mask"].sum().item(),
            0 if teacher_input is None else teacher_input["attention_mask"].sum().item(),
        )
        self.tokens_seen += num_tokens
        self.log("tokens_seen", self.tokens_seen, on_step=True, reduce_fx="max")  # We sum it up ourselves
        self.log("loss", losses["total"], prog_bar=True)
        self.log_dict(losses)
        # ####
        # # Log additional metrics for more insight into the training
        # if self.log_additional_metrics:
        #     with torch.autocast(device_type="cuda", enabled=False):
        #         # cossim = normalize(embs[:, 0]) @ normalize(embt[:, 0]).T
        #         crossentropy = torch.nn.functional.cross_entropy(
        #             (embs[:, 0] @ embt[:, 0].T) / math.sqrt(embs.shape[-1]),
        #             torch.arange(embs.shape[0], device=self.device),
        #         )
        #         # Equivalent to above: -torch.diag(torch.log_softmax(embs[:, 0] @ embt[:, 0].T, dim=, -1)).mean()
        #         # crosscorr = normalize(embs[:, 0], dim=0).T @ normalize(embt[:, 0], dim=0) / embt.shape[0]
        #     metrics = {
        #         "crossentropy": crossentropy,
        #         # "pairwise-cossim": cossim,
        #         # "crosscorrelation": crosscorr,
        #     }
        #     for metric_name, metric_value in metrics.items():
        #         if metric_value.ndim > 1:
        #             if self.logger is not None and (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
        #                 self.logger.log_image(metric_name, [metric_value])
        #         else:
        #             self.log(metric_name, metric_value)
        # ####
        return losses["total"]

    def save_pretrained(self, save_path: str | Path) -> None:
        self.student.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def on_save_checkpoint(self, *args, **kwargs) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.save_pretrained(save_path)
