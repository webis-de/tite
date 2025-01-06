from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities import grad_norm
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerBase

from .datasets import GLUEDataModule, IRDatasetsDataModule
from .glue_module import GlueModule
from .jepa import JEPA, LossFn
from .model import TiteConfig, TiteModel
from .msmarco_module import MSMARCOModule
from .predictor import MAEDecoder, MAEEnhancedDecoder, MLMDecoder


class _DetachFromGrad(Module):
    def __init__(self, module: Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = module

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            output = self.module(*args, **kwargs)
        assert isinstance(output, Tensor)
        return output.detach()  # Better safe than sorry


class ComposedEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        small_embedding_dim: int,
        large_embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            large_embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )
        self.linear = None
        if small_embedding_dim != large_embedding_dim:
            self.linear = torch.nn.Linear(large_embedding_dim, small_embedding_dim, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        embeddings = torch.nn.functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        if self.linear is not None:
            embeddings = self.linear(embeddings)
        return embeddings


def tie_weights(student_embedding: torch.nn.Embedding, teacher_weight: torch.nn.Parameter) -> torch.nn.Embedding:
    if student_embedding.weight.data.shape == teacher_weight.data.shape:
        student_embedding.weight.data = teacher_weight.data
        return student_embedding
    composed_embedding = ComposedEmbedding(
        student_embedding.num_embeddings,
        student_embedding.embedding_dim,
        teacher_weight.data.shape[1],
        student_embedding.padding_idx,
        _weight=teacher_weight.data,
    ).to(student_embedding.weight.device)
    composed_embedding.weight.data = teacher_weight.data
    return composed_embedding


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

    def on_train_start(self) -> None:
        # tie weights
        decoder = None
        position_embeddings = None
        for predictor in self.predictors:
            if isinstance(predictor, (MLMDecoder, MAEDecoder, MAEEnhancedDecoder)):
                if decoder is None:
                    decoder = predictor.decoder
                else:
                    predictor.decoder = decoder
                if getattr(predictor, "embeddings", None) is not None:
                    predictor.embeddings.word_embeddings.weight.data = decoder.weight.data
                    if predictor.embeddings.position_embeddings is not None:
                        if position_embeddings is None:
                            position_embeddings = predictor.embeddings.position_embeddings
                        else:
                            predictor.embeddings.position_embeddings = position_embeddings

        self.student.embeddings.word_embeddings = tie_weights(self.student.embeddings.word_embeddings, decoder.weight)
        if position_embeddings is not None and self.student.embeddings.position_embeddings is not None:
            self.student.embeddings.position_embeddings = tie_weights(
                self.student.embeddings.position_embeddings, position_embeddings.weight
            )
        assert all(decoder.weight.data_ptr() == predictor.decoder.weight.data_ptr() for predictor in self.predictors)
        assert all(
            getattr(predictor, "embeddings", None) is None
            or decoder.weight.data_ptr() == predictor.embeddings.word_embeddings.weight.data_ptr()
            for predictor in self.predictors
        )
        assert all(
            getattr(predictor, "embeddings", None) is None
            or predictor.embeddings.position_embeddings is None
            or position_embeddings.weight.data_ptr() == predictor.embeddings.position_embeddings.weight.data_ptr()
            for predictor in self.predictors
        )
        assert self.student.embeddings.word_embeddings.weight.data_ptr() == decoder.weight.data_ptr()
        if self.student.embeddings.position_embeddings is not None and position_embeddings is not None:
            assert (
                self.student.embeddings.position_embeddings.weight.data_ptr() == position_embeddings.weight.data_ptr()
            )

    def on_validation_start(self) -> None:
        if self.trainer is None:
            return
        if self.trainer.limit_val_batches == 0:
            return
        add_special_tokens = self.trainer.datamodule.collator.add_special_tokens
        enable_progress_bar = self.trainer.progress_bar_callback is not None
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
                    enable_checkpointing=False,
                    num_sanity_val_steps=0,
                    enable_progress_bar=enable_progress_bar,
                    limit_train_batches=2 if self.trainer is not None and self.trainer.sanity_checking else None,
                    limit_val_batches=2 if self.trainer is not None and self.trainer.sanity_checking else None,
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
            max_steps = 5_000
            trainer = Trainer(
                logger=False,
                precision=(self.trainer.precision if self.trainer is not None else "bf16-mixed"),
                max_steps=max_steps,
                max_epochs=1,
                enable_checkpointing=False,
                num_sanity_val_steps=0,
                val_check_interval=2 if self.trainer is not None and self.trainer.sanity_checking else max_steps,
                enable_progress_bar=enable_progress_bar,
                limit_train_batches=2 if self.trainer is not None and self.trainer.sanity_checking else None,
                limit_val_batches=2 if self.trainer is not None and self.trainer.sanity_checking else None,
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
        losses, output = self.jepa(student_input, teacher_input, **batch)
        losses["total"] = sum(losses.values())
        num_tokens = max(
            student_input["attention_mask"].sum().item(),
            0 if teacher_input is None else teacher_input["attention_mask"].sum().item(),
        )
        self.tokens_seen += num_tokens
        self.log("tokens_seen", self.tokens_seen, on_step=True, reduce_fx="max")  # We sum it up ourselves
        self.log("loss", losses["total"], prog_bar=True)
        self.log_dict(losses)
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
