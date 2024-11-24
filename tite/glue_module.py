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
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.num_classes = NUM_CLASSES_MAP[task]
        last_dim = 768
        layers = []
        hidden_size = model.config.last_hidden_size
        while True:
            if hidden_size == last_dim:
                break
            new_hidden_size = max(hidden_size // 2, last_dim)
            layers.append(torch.nn.Linear(hidden_size, new_hidden_size))
            layers.append(torch.nn.Dropout(0.1))
            layers.append(torch.nn.ReLU())
            hidden_size = new_hidden_size
        layers.append(torch.nn.Linear(last_dim, self.num_classes))
        self.classification_head = torch.nn.Sequential(*layers)
        if self.num_classes == 1:
            self.loss_function = torch.nn.MSELoss()
        else:
            self.loss_function = torch.nn.CrossEntropyLoss()
        self.evaluation_metrics = torch.nn.ModuleList(METRICS_MAP[task])

    def training_step(self, batch) -> Tensor:
        output = self.model(batch["input_ids"], batch["attention_mask"]).last_hidden_state
        logits = self.classification_head(output[:, 0])
        loss = self.loss_function(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, *args, **kwargs) -> None:
        output = self.model(batch["input_ids"], batch["attention_mask"])
        logits = self.classification_head(output.last_hidden_state[:, 0])
        if self.num_classes > 1:
            logits = logits.argmax(-1)
        for validation_metric in self.evaluation_metrics:
            metric = validation_metric(logits, batch["label"])
            self.log(validation_metric.__class__.__name__, metric)

    def test_step(self, batch: dict[str, Any], *args, **kwargs) -> None:
        self.validation_step(batch, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return opt
