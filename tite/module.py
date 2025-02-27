import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities import grad_norm
from transformers import BatchEncoding, PreTrainedTokenizerBase

from .datasets import GLUEDataModule, IRDatasetsDataModule
from .decoder import Decoder
from .glue_module import GlueModule
from .loss import LossFunction
from .model.tite import ComposedEmbedding, TiteModel
from .msmarco_module import MSMARCOModule
from .teacher import Teacher


def _parse_kwargs(kwargs: dict[str, Any], module: Teacher | Decoder | TiteModel) -> dict[str, Any]:
    if isinstance(module, Teacher):
        valid_keys = inspect.signature(module.map_targets).parameters.keys()
    elif isinstance(module, Decoder | TiteModel):
        valid_keys = inspect.signature(module.forward).parameters.keys()
    else:
        raise ValueError(f"Unsupported module type {type(module)}")
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def tie_weights(encoder_embedding: ComposedEmbedding, decoder_weight: torch.nn.Parameter) -> ComposedEmbedding:
    if encoder_embedding.weight.data.shape == decoder_weight.data.shape:
        encoder_embedding.weight.data = decoder_weight.data
        return encoder_embedding
    raise NotImplementedError("Make sure this works for composed embeddings")
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
        decoders: list[Decoder],
        teachers: list[Teacher],
        loss_functions: list[LossFunction],
        log_additional_metrics: bool = False,
        validate_on_glue: bool = False,
        validate_on_trec_dl: bool = False,
        log_gradients: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.decoders = torch.nn.ModuleList(decoders)
        self.teachers = teachers
        self.loss_functions = torch.nn.ModuleList(loss_functions)

        self.log_additional_metrics = log_additional_metrics
        self.validate_on_glue = validate_on_glue
        self.validate_on_trec_dl = validate_on_trec_dl
        self.log_gradients = log_gradients

        self.tokens_seen = torch.tensor(0.0)

    def on_train_start(self) -> None:
        # tie weights
        # first unify the decoder weights and embeddings for the decoders
        decoding_layer: torch.nn.Linear | None = None
        position_embeddings: torch.nn.Embedding | None = None
        for decoder in self.decoders:
            if not hasattr(decoder, "decoder"):
                continue
            if decoding_layer is None:
                decoding_layer = getattr(decoder, "decoder", None)
            setattr(decoder, "decoder", decoding_layer)
            embeddings = getattr(decoder, "embeddings", None)
            if embeddings is not None:
                assert decoding_layer is not None
                embeddings.word_embeddings.weight.data = decoding_layer.weight.data
                if embeddings.position_embeddings is not None:
                    if position_embeddings is None:
                        position_embeddings = embeddings.position_embeddings
                    else:
                        embeddings.position_embeddings = position_embeddings

        # then tie the unified decoder weights and embeddings to the encoder
        assert decoding_layer is not None
        self.encoder.embeddings.word_embeddings = tie_weights(
            self.encoder.embeddings.word_embeddings, decoding_layer.weight
        )
        if position_embeddings is not None and self.encoder.embeddings.position_embeddings is not None:
            self.encoder.embeddings.position_embeddings = tie_weights(
                self.encoder.embeddings.position_embeddings, position_embeddings.weight
            )
        self.assert_same_weights()

    def assert_same_weights(self):
        assert all(
            self.encoder.embeddings.word_embeddings.weight.data_ptr() == decoder.decoder.weight.data_ptr()
            for decoder in self.decoders
        )
        assert all(
            getattr(decoder, "embeddings", None) is None
            or self.encoder.embeddings.word_embeddings.weight.data_ptr()
            == decoder.embeddings.word_embeddings.weight.data_ptr()
            for decoder in self.decoders
        )
        assert all(
            getattr(decoder, "embeddings", None) is None
            or decoder.embeddings.position_embeddings is None
            or self.encoder.embeddings.position_embeddings is None
            or (
                self.encoder.embeddings.position_embeddings.weight.data_ptr()
                == decoder.embeddings.position_embeddings.weight.data_ptr()
            )
            for decoder in self.decoders
        )

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
        enable_progress_bar = self.trainer.progress_bar_callback is not None
        msmarco = IRDatasetsDataModule(
            tokenizer=self.tokenizer,
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
        for name, module in [("encoder", self.encoder)] + [
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

    def training_step(self, batch: list[tuple[BatchEncoding, dict]]) -> torch.Tensor:
        encoder_encoding, encoder_auxiliary_data = batch[0]
        decoder_inputs = batch[1:]

        encoder_kwargs = _parse_kwargs({**encoder_encoding, **encoder_auxiliary_data}, self.encoder)
        encoder_output = self.encoder(**encoder_kwargs).last_hidden_state
        losses = {}
        for decoder_input, decoder, teacher, loss_function in zip(
            decoder_inputs, self.decoders, self.teachers, self.loss_functions
        ):
            decoder_encoding, decoder_auxiliary_data = decoder_input
            decoder_kwargs = _parse_kwargs(
                {
                    "encoder_output": encoder_output,
                    **encoder_auxiliary_data,
                    **decoder_encoding,
                    **decoder_auxiliary_data,
                },
                decoder,
            )
            decoder_output = decoder(**decoder_kwargs)
            teacher_kwargs = _parse_kwargs(
                {**encoder_auxiliary_data, **decoder_auxiliary_data, **decoder_encoding}, teacher
            )
            target = teacher(**teacher_kwargs)
            losses[loss_function.__class__.__name__] = loss_function(decoder_output, target)
        losses["total"] = sum(losses.values())
        num_tokens = encoder_encoding["attention_mask"].sum()
        self.tokens_seen += num_tokens.to(self.tokens_seen.device)
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
