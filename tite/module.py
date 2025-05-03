from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities import grad_norm
from transformers import PreTrainedTokenizerBase

from .datasets import GLUEDataModule, IRDatasetsDataModule
from .glue_module import GlueModule
from .model.tite import TiteForPreTraining
from .msmarco_module import MSMARCOModule


class TiteModule(LightningModule):
    def __init__(
        self,
        model: TiteForPreTraining,
        tokenizer: PreTrainedTokenizerBase,
        validate_on_glue: bool = False,
        validate_on_trec_dl: bool = False,
        log_gradients: bool = False,
        compile: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        if compile:
            self.model.compile(dynamic=True)
        self.tokenizer = tokenizer

        self.validate_on_glue = validate_on_glue
        self.validate_on_trec_dl = validate_on_trec_dl
        self.log_gradients = log_gradients

        self.tokens_seen = torch.tensor(0.0)

    def _validate_on_glue(self):
        enable_progress_bar = self.trainer.progress_bar_callback is not None
        # for task in TASK_COLUMN_NAMES:
        metrics = {}
        for task in ["mrpc"]:
            glue = GLUEDataModule(
                task=task,
                tokenizer=self.tokenizer,
                batch_size=32,
                streaming=False,
            )
            copy_model = deepcopy(self.model).train()
            if hasattr(copy_model.config, "pooling") and getattr(copy_model.config, "pooling") is None:
                copy_model.config.pooling = "first"
            glue_module = GlueModule(copy_model, self.tokenizer, glue.hparams.name)
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
            for name, value in trainer.logged_metrics.items():
                if "step" in name:
                    continue
                metrics[f"{glue.hparams.name}/{name}"] = value
        return metrics

    def on_train_start(self) -> None:
        self.tokens_seen = self.tokens_seen.to(self.device)
        return super().on_train_start()

    def _validate_on_trec_dl(self):
        enable_progress_bar = self.trainer.progress_bar_callback is not None
        msmarco = IRDatasetsDataModule(
            tokenizer=self.tokenizer,
            trainset=("msmarco-passage/train/triples-small", "triples"),
            valset=("msmarco-passage/trec-dl-2019/judged", "scoreddocs"),
            batch_size=32,
            inference_batch_size=256,
        )
        copy_model = deepcopy(self.model).train()
        if hasattr(copy_model.config, "pooling") and getattr(copy_model.config, "pooling") is None:
            copy_model.config.pooling = "first"
        msmarco_module = MSMARCOModule(copy_model, self.tokenizer)
        max_steps = 10_000
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
        metrics = {}
        for name, value in trainer.logged_metrics.items():
            if "step" in name:
                continue
            metrics[f"trec-dl-2019/{name}"] = value
        return metrics

    def on_validation_start(self) -> None:
        if self.trainer is None:
            return
        if self.trainer.limit_val_batches == 0:
            return
        if self.validate_on_glue:
            metrics = self._validate_on_glue()
            for name, value in metrics.items():
                self.log(name, value, on_step=False, on_epoch=True)
        if self.validate_on_trec_dl:
            metrics = self._validate_on_trec_dl()
            for name, value in metrics.items():
                self.log(name, value, on_step=False, on_epoch=True)

    def on_before_optimizer_step(self, optimizer):
        if not self.log_gradients:
            return
        for name, module in [("encoder", self.model)] + [
            (f"decoder_{idx}", decoder) for idx, decoder in enumerate(self.decoders)
        ]:
            norms = grad_norm(module, norm_type=2)
            if not norms:
                continue
            total_norm = norms["grad_2.0_norm_total"]
            module_norms = {f"{name}_grad_2.0_norm_total": total_norm}
            if self.log_gradients:
                self.log_dict(module_norms)

    def validation_step(self, batch: dict[str, Any] | None) -> None:
        # Empty validation step to trick pytorch lightning into validating this model though validation is actually done
        # using the GlueModule and MSMARCOModule
        return

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        encoding, transformed_encoding, auxiliary_data = batch
        labels = {}
        for name, head in self.model.heads.items():
            labels[name] = head.get_labels(**encoding, **auxiliary_data)

        output = self.model.forward(**transformed_encoding, original_input_ids=encoding["input_ids"], labels=labels)

        losses = output.losses
        assert losses is not None
        losses["total"] = sum(losses.values())
        num_tokens = encoding["attention_mask"].sum()
        self.tokens_seen += num_tokens
        self.log("tokens_seen", self.tokens_seen, on_step=True, reduce_fx="max")  # We sum it up ourselves
        self.log("loss", losses["total"], prog_bar=True)
        self.log_dict(losses)
        return losses["total"]

    def save_pretrained(self, save_path: str | Path) -> None:
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def on_save_checkpoint(self, *args, **kwargs) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.save_pretrained(save_path)
