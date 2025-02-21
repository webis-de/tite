import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities import grad_norm
from transformers import PreTrainedTokenizerBase

from .datasets import GLUEDataModule, IRDatasetsDataModule
from .glue_module import GlueModule
from .model import TiteModel
from .msmarco_module import MSMARCOModule


def _parse_kwargs(kwargs: dict[str, Any], module: torch.nn.Module) -> dict[str, Any]:
    valid_keys = inspect.signature(module.forward).parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def tie_weights(encoder_embedding: torch.nn.Embedding, decoder_weight: torch.nn.Parameter) -> torch.nn.Embedding:
    if encoder_embedding.weight.data.shape == decoder_weight.data.shape:
        encoder_embedding.weight.data = decoder_weight.data
        return encoder_embedding
    composed_embedding = ComposedEmbedding(
        encoder_embedding.num_embeddings,
        encoder_embedding.embedding_dim,
        decoder_weight.data.shape[1],
        encoder_embedding.padding_idx,
        _weight=decoder_weight.data,
    ).to(encoder_embedding.weight.device)
    composed_embedding.weight.data = decoder_weight.data
    return composed_embedding


class TiteModule(LightningModule):
    def __init__(
        self,
        encoder: TiteModel,
        tokenizer: PreTrainedTokenizerBase,
        decoders: list[torch.nn.Module],
        predictors: list[torch.nn.Module],
        losses: list[torch.nn.Module],
        log_additional_metrics: bool = False,
        validate_on_glue: bool = False,
        validate_on_trec_dl: bool = False,
        log_gradients: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.decoders = torch.nn.ModuleList([decoder if decoder is not None else encoder for decoder in decoders])
        self.predictors = torch.nn.ModuleList(predictors)
        self.losses = torch.nn.ModuleList(losses)

        self.log_additional_metrics = log_additional_metrics
        self.validate_on_glue = validate_on_glue
        self.validate_on_trec_dl = validate_on_trec_dl
        self.log_gradients = log_gradients

        self.tokens_seen = 0.0

    def on_train_start(self) -> None:
        # tie weights
        # first unify the decoder weights and embeddings for the predictors
        decoder: torch.nn.Linear | None = None
        position_embeddings: torch.nn.Embedding | None = None
        for predictor in self.predictors:
            if not hasattr(predictor, "decoder"):
                continue
            if decoder is None:
                decoder = getattr(predictor, "decoder", None)
            setattr(predictor, "decoder", decoder)
            embeddings = getattr(predictor, "embeddings", None)
            if embeddings is not None:
                assert decoder is not None
                embeddings.word_embeddings.weight.data = decoder.weight.data
                if embeddings.position_embeddings is not None:
                    if position_embeddings is None:
                        position_embeddings = embeddings.position_embeddings
                    else:
                        embeddings.position_embeddings = position_embeddings

        # then tie the unified decoder weights and embeddings to the encoder
        assert decoder is not None
        self.encoder.embeddings.word_embeddings = tie_weights(self.encoder.embeddings.word_embeddings, decoder.weight)
        if position_embeddings is not None and self.encoder.embeddings.position_embeddings is not None:
            self.encoder.embeddings.position_embeddings = tie_weights(
                self.encoder.embeddings.position_embeddings, position_embeddings.weight
            )
        self.assert_same_weights()

    def assert_same_weights(self):
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
        assert self.encoder.embeddings.word_embeddings.weight.data_ptr() == decoder.weight.data_ptr()
        if self.encoder.embeddings.position_embeddings is not None and position_embeddings is not None:
            assert (
                self.encoder.embeddings.position_embeddings.weight.data_ptr() == position_embeddings.weight.data_ptr()
            )

    def _validate_on_glue(self):
        add_special_tokens = self.trainer.datamodule.collator.add_special_tokens
        enable_progress_bar = self.trainer.progress_bar_callback is not None
        # for task in TASK_COLUMN_NAMES:
        metrics = {}
        for task in ["mrpc"]:
            glue = GLUEDataModule(
                task=task,
                tokenizer=self.tokenizer,
                batch_size=32,
                add_special_tokens=add_special_tokens,
                streaming=False,
            )
            copy_encoder = deepcopy(self.encoder).train()
            if hasattr(copy_encoder.config, "pooling") and getattr(copy_encoder.config, "pooling") is None:
                copy_encoder.config.pooling = "first"
            glue_module = GlueModule(copy_encoder, self.tokenizer, glue.hparams.name)
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

    def _validate_on_trec_dl(self):
        add_special_tokens = self.trainer.datamodule.collator.add_special_tokens
        enable_progress_bar = self.trainer.progress_bar_callback is not None
        msmarco = IRDatasetsDataModule(
            tokenizer=self.tokenizer,
            add_special_tokens=add_special_tokens,
            trainset=("msmarco-passage/train/triples-small", "triples"),
            valset=("msmarco-passage/trec-dl-2019/judged", "scoreddocs"),
            batch_size=32,
            inference_batch_size=256,
        )
        copy_encoder = deepcopy(self.encoder).train()
        if hasattr(copy_encoder.config, "pooling") and getattr(copy_encoder.config, "pooling") is None:
            copy_encoder.config.pooling = "first"
        msmarco_module = MSMARCOModule(copy_encoder, self.tokenizer)
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
        for name, module in (
            [("encoder", self.encoder)]
            + [(f"decoder_{idx}", decoder) for idx, decoder in enumerate(self.decoders)]
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

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoder_input = batch.pop("encoder_input")
        decoder_input = batch.pop("decoder_input", None)

        encoder_aux = {k[8:]: v for k, v in batch.items() if k.startswith("encoder_")}
        decoder_aux = {k[8:]: v for k, v in batch.items() if k.startswith("decoder_")}
        encoder_output = self.encoder(**encoder_input).last_hidden_state
        losses = {}
        for decoder, predictor, loss in zip(self.decoders, self.predictors, self.losses):
            if decoder_input is None or isinstance(decoder, CopyEncoder):
                decoder_output = encoder_output.detach()
            else:
                decoder_kwargs = _parse_kwargs({**encoder_aux, **decoder_aux}, decoder)
                decoder_output = decoder(**target, **decoder_kwargs)
                if isinstance(decoder_output, TiteModelOutput):
                    decoder_output = decoder_output.last_hidden_state
            predictor_kwargs = _parse_kwargs({**encoder_aux, **decoder_aux, **(target or {})}, predictor)
            pred = predictor(encoder_output, **predictor_kwargs)
            losses[loss.__class__.__name__] = loss(pred, emby)
        losses["total"] = sum(losses.values())
        num_tokens = max(
            encoder_input["attention_mask"].sum().item(),
            0 if decoder_input is None else decoder_input["attention_mask"].sum().item(),
        )
        self.tokens_seen += num_tokens
        self.log("tokens_seen", self.tokens_seen, on_step=True, reduce_fx="max")  # We sum it up ourselves
        self.log("loss", losses["total"], prog_bar=True)
        self.log_dict(losses)
        return losses["total"]

    def save_pretrained(self, save_path: str | Path) -> None:
        self.encoder.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def on_save_checkpoint(self, *args, **kwargs) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.save_pretrained(save_path)
