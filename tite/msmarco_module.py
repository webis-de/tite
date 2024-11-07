from typing import Any

import ir_datasets
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
        self.model = model
        self.tokenizer = tokenizer
        self.validation_step_outputs = []
        self.ndcgs = []

    def training_step(self, batch) -> Tensor:
        query_emb = self.model(**batch["encoded_query"])
        pos_doc_emb = self.model(**batch["encoded_pos_doc"])
        neg_doc_emb = self.model(**batch["encoded_neg_doc"])
        # pos_sim = (query_emb @ pos_doc_emb.transpose(-1, -2)).view(-1)
        # neg_sim = (query_emb @ neg_doc_emb.transpose(-1, -2)).view(-1)
        pos_sim = torch.nn.functional.cosine_similarity(query_emb.squeeze(1), pos_doc_emb.squeeze(1))
        neg_sim = torch.nn.functional.cosine_similarity(query_emb.squeeze(1), neg_doc_emb.squeeze(1))
        margin = pos_sim - neg_sim
        loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, *args, **kwargs) -> None:
        query_emb = self.model(**batch["encoded_query"])
        doc_emb = self.model(**batch["encoded_doc"])
        sim = (query_emb @ doc_emb.transpose(-1, -2)).view(-1)
        self.validation_step_outputs.extend(zip(batch["query_id"], batch["doc_id"], sim.tolist(), batch["dataset_id"]))
        # logits = self.classification_head(output.mean(1))

    def on_validation_epoch_end(self) -> None:
        df = pd.DataFrame(self.validation_step_outputs, columns=["query_id", "doc_id", "score", "dataset_id"])
        self.validation_step_outputs.clear()
        assert df["dataset_id"].drop_duplicates().count() == 1
        dataset_id = df["dataset_id"].iloc[0]
        run = df.drop(columns=["dataset_id"])
        dataset = ir_datasets.load(dataset_id)
        qrels = pd.DataFrame(dataset.qrels_iter())
        measure = ir_measures.parse_measure("nDCG@10")
        value = measure.calc_aggregate(qrels, run)
        self.ndcgs.append(value)
        self.log("nDCG@10", value)

    def test_step(self, batch: dict[str, Any], *args, **kwargs) -> None:
        self.validation_step(batch, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return opt
