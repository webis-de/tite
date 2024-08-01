from functools import reduce
from typing import Any, Callable, Iterable, Mapping, NamedTuple

import torch
from lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import default_collate
from transformers import PreTrainedTokenizerBase, TensorType


class MaskCollatorBatch(NamedTuple):
    input_ids: Tensor
    ctxt_attn_mask: Tensor
    pred_attn_mask: Tensor
    metadata: dict[str, Any]


Encoder = Callable[[Any], Any]
Predictor = Callable[[Any, Any], Any]
LossFn = Callable[[Any, Any], Any]


class JEPA(Module):
    def __init__(
        self,
        student: Encoder,
        teacher: Encoder,
        predictor: Predictor,
        loss: LossFn,
        return_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self._student = student
        self._teacher = teacher
        self._predictor = predictor
        self._loss = loss
        self._return_embeddings = return_embeddings

    def forward(self, input: dict, target: dict, aux: Any | None = None):
        embx = self._student(**input)
        emby = self._teacher(
            **{f"student_{k}": v for k, v in input.items()}, **{f"teacher_{k}": v for k, v in target.items()}
        )
        pred = self._predictor(embx, aux)
        if not self._return_embeddings:
            return self._loss(pred, emby)
        return self._loss(pred, emby), embx, emby


class TJEPA(LightningModule):
    """JEPA adapted for text"""

    def __init__(self, student: Module, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.student = student

    def forward(self, *args, **kwargs):
        self.student.forward(*args, **kwargs)

    def training_step(self, batch: MaskCollatorBatch) -> Tensor | Mapping[str, Any] | None:
        print(batch.ctxt_attn_mask)
        print(batch.pred_attn_mask)
        raise NotImplementedError()

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.student.parameters())


class MaskCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_key: str = "text",
        meta_keys: list[str] = [],
        num_pred_blocks: int = 4,
        ctxt_mask_ratio: tuple[float, float] = (0.85, 1.0),
        pred_mask_ratio: tuple[float, float] = (0.15, 0.2),
    ) -> None:
        self._tokenizer = tokenizer
        self._text_key = text_key
        self._meta_keys = meta_keys
        self._num_pred_blocks = num_pred_blocks
        self._ctxt_mask_ratio = ctxt_mask_ratio
        self._pred_mask_ratio = pred_mask_ratio

    @staticmethod
    def _to_mask(idx_min: Tensor, idx_max: Tensor, seqlen: Tensor) -> Tensor:
        """Creates a mask vector from the min (inclusive) and max (exclusive) indices.

        Args:
            idx_min (Tensor): The first index to keep
            idx_max (Tensor): The first index not to keep
            seqlen (Tensor): The length of each sequence

        Returns:
            Tensor: A mask tensor masking everything except the specified index range.
        """
        maxseq = torch.max(seqlen)
        idx = torch.arange(maxseq).unsqueeze(0).expand(len(idx_min), -1)
        return torch.logical_and(idx >= idx_min.unsqueeze(-1), idx < idx_max.unsqueeze(-1))

    @staticmethod
    def _sample_chunks(scale: tuple[float, float], seqlen: Tensor) -> tuple[Tensor, Tensor]:
        """Samples a chunk for each entry in `seqlen`.

        Args:
            scale (tuple[float, float]): The minimum and maximum ratio of the chunk's size to the sequence.
            seqlen (Tensor): The length of each sequence.

        Returns:
            tuple[Tensor, Tensor]: A chunk for each sequence in the form (start_idx, end_idx).
        """
        num = len(seqlen)
        min, max = scale
        rel_start = torch.rand(num)
        rel_length = min + (max - min) * torch.rand(num)
        chunk_len = torch.round(rel_length * seqlen)
        chunk_start = torch.round(rel_start * (seqlen - chunk_len))
        return chunk_start, chunk_start + chunk_len

    def _sample_mask(self, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        # 1) Compute the length of each input sequence
        seqlen = attn_mask.sum(dim=1)  # NOTE: we assume here that attn_mask contains only 1s for each input token
        # 2) Sample `self._num_pred_blocks` prediction blocks
        pred_blocks = torch.stack(
            [
                MaskCollator._to_mask(*MaskCollator._sample_chunks(self._pred_mask_ratio, seqlen), seqlen)
                for _ in range(self._num_pred_blocks)
            ]
        )
        # 3) Sample a single context block
        ctxt_block = MaskCollator._to_mask(*MaskCollator._sample_chunks(self._ctxt_mask_ratio, seqlen), seqlen)
        # 4) Remove prediction blocks from the context block
        not_pred = reduce(torch.logical_or, pred_blocks).logical_not()
        ctxt_block = attn_mask.logical_and(ctxt_block.logical_and(not_pred))
        # TODO: Maybe check if the context block still contains a reasonable amount of information
        return ctxt_block, pred_blocks

    def __call__(self, batch: Iterable[dict[str, Any]]) -> MaskCollatorBatch:
        """Tokenizes and collates the input batch.

        Args:
            batch (Iterable[dict[str, Any]]): An iterable containing instances of the dataset. Use the text_key
            constructor argument to select the appropriate value for the passage. And meta_keys to (optionally) include
            meta information.

        Returns:
            tuple[Tensor, Tensor, Tensor, list[Any]]: A quadrupel of the collated token ids, context attention mask,
            prediction attention mask, and meta information
        """
        texts, meta = zip(*((d[self._text_key], [d[k] for k in self._meta_keys]) for d in batch))
        tokenized = self._tokenizer(
            text=texts,
            return_attention_mask=True,
            padding=True,
            return_tensors=TensorType.PYTORCH,
        )
        metadata = default_collate(meta)
        ctxt_attn, pred_attn = self._sample_mask(tokenized["attention_mask"])
        return MaskCollatorBatch(
            input_ids=tokenized["input_ids"],
            ctxt_attn_mask=ctxt_attn,
            pred_attn_mask=pred_attn,
            metadata=metadata,
        )
