from typing import Any

import torch
import torchmetrics
import torchmetrics.classification
from lightning import LightningModule
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .model import TiteModel

NUM_CLASSES_MAP = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "stsb": 1,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    # "wnli": 2, # excluded in MosaicBERT and original BERT
}
METRICS_MAP = {
    "cola": [torchmetrics.MatthewsCorrCoef(num_classes=2, task="multiclass")],
    "sst2": [torchmetrics.classification.MulticlassAccuracy(num_classes=2, average="micro")],
    "mrpc": [
        torchmetrics.classification.BinaryF1Score(),
        torchmetrics.classification.MulticlassAccuracy(num_classes=2, average="micro"),
        torchmetrics.MatthewsCorrCoef(num_classes=2, task="multiclass"),
    ],
    "stsb": [torchmetrics.PearsonCorrCoef()],
    "qqp": [
        torchmetrics.classification.BinaryF1Score(),
        torchmetrics.classification.MulticlassAccuracy(num_classes=2, average="micro"),
    ],
    "mnli": [torchmetrics.classification.MulticlassAccuracy(num_classes=3, average="micro")],
    "qnli": [torchmetrics.classification.MulticlassAccuracy(num_classes=2, average="micro")],
    "rte": [torchmetrics.classification.MulticlassAccuracy(num_classes=2, average="micro")],
}


class GlueModule(LightningModule):
    def __init__(self, model: TiteModel, tokenizer: PreTrainedTokenizerBase, task: str) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._task = task
        self.num_classes = NUM_CLASSES_MAP[task]
        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(model.config.last_hidden_size, self.num_classes),
        )
        if self.num_classes == 1:
            self._loss_function = torch.nn.MSELoss()
        else:
            self._loss_function = torch.nn.CrossEntropyLoss()
        self._evaluation_metrics = torch.nn.ModuleList(METRICS_MAP[task])

    def training_step(self, batch) -> Tensor:
        tokenized = {"input_ids": batch["teacher_input_ids"], "attention_mask": batch["teacher_attention_mask"]}
        output = self._model(**tokenized)
        # logits = self.classification_head(output.mean(1))
        logits = self.classification_head(output[:, 0])
        loss = self._loss_function(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, *args, **kwargs) -> None:
        tokenized = {"input_ids": batch["teacher_input_ids"], "attention_mask": batch["teacher_attention_mask"]}
        output = self._model(**tokenized)
        # logits = self.classification_head(output.mean(1))
        logits = self.classification_head(output[:, 0])
        if self.num_classes > 1:
            logits = logits.argmax(-1)
        for validation_metric in self._evaluation_metrics:
            metric = validation_metric(logits, batch["label"])
            self.log(validation_metric.__class__.__name__, metric)

    def test_step(self, batch: dict[str, Any], *args, **kwargs) -> None:
        self.validation_step(batch, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return opt
