from typing import Any

import ir_measures
import pandas as pd
import torch
from lightning import LightningModule
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .model import TiteModel


class MSMARCOModule(LightningModule):
    def __init__(self, model: TiteModel, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._validation_step_outputs = []

    def training_step(self, batch) -> Tensor:
        query_ids, query_encoding, pos_doc_ids, pos_doc_encoding, neg_doc_ids, neg_doc_encoding = batch
        query_emb = self._model(**query_encoding)
        pos_doc_emb = self._model(**pos_doc_encoding)
        neg_doc_emb = self._model(**neg_doc_encoding)
        pos_sim = (query_emb @ pos_doc_emb.transpose(-1, -2)).view(-1)
        neg_sim = (query_emb @ neg_doc_emb.transpose(-1, -2)).view(-1)
        margin = pos_sim - neg_sim
        loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, *args, **kwargs) -> None:
        query_ids, query_encoding, doc_ids, doc_encoding, _, rel = batch
        query_emb = self._model(**query_encoding)
        doc_emb = self._model(**doc_encoding)
        sim = (query_emb @ doc_emb.transpose(-1, -2)).view(-1)
        self._validation_step_outputs.extend(zip(query_ids, doc_ids, sim.tolist(), rel))
        # logits = self.classification_head(output.mean(1))

    def on_validation_epoch_end(self) -> None:
        qid, did, sim, rel = zip(*self._validation_step_outputs)
        qrels = pd.DataFrame({"query_id": qid, "doc_id": did, "relevance": rel})
        run = pd.DataFrame({"query_id": qid, "doc_id": did, "score": sim})
        measure = ir_measures.parse_measure("nDCG@10")
        value = measure.calc_aggregate(qrels, run)
        self.log("nDCG@10", value)

    def test_step(self, batch: dict[str, Any], *args, **kwargs) -> None:
        self.validation_step(batch, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return opt
