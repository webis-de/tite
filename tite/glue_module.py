from typing import Any

import torch
import torchmetrics
import torchmetrics.classification
from lightning import LightningModule
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerBase, TensorType, get_constant_schedule_with_warmup

from .model import TiteModel

NUM_CLASSES_MAP = {
    "mnli": 3,
    "rte": 2,
    "qqp": 2,
    "cola": 2,
    "mrpc": 2,
    "qnli": 2,
    "sst2": 2,
    "stsb": 1,
}
TASK_COLUMN_NAMES = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}


LOSS_FUNCTION_MAP = {}


class GlueModule(LightningModule):
    def __init__(self, model: TiteModel, tokenizer: PreTrainedTokenizerBase, task: str) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._task = task
        num_classes = NUM_CLASSES_MAP[task]
        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(model.config.hidden_size[-1], num_classes),
        )
        if num_classes == 1:
            self._loss_function = torch.nn.MSELoss()
            self._evaluation_metrics = torch.nn.ModuleList(
                [torchmetrics.MeanSquaredError(), torchmetrics.PearsonCorrCoef()]
            )
        else:
            self._loss_function = torch.nn.CrossEntropyLoss()
            self._evaluation_metrics = torch.nn.ModuleList(
                [
                    torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average="micro"),
                    torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=num_classes),
                ]
            )
            if num_classes == 2:
                self._evaluation_metrics.append(
                    torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, average="micro")
                )

    def parse_batch(self, batch: dict[str, Any]) -> BatchEncoding:
        text_column_names = TASK_COLUMN_NAMES[self._task]
        first_half = batch[text_column_names[0]]
        second_half = batch[text_column_names[1]] if text_column_names[1] in batch else None
        encoded = self._tokenizer(
            first_half,
            second_half,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            return_tensors=TensorType.PYTORCH,
            truncation=True,
        )
        encoded = encoded.to(self.device)
        return encoded

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        encoded = self.parse_batch(batch)
        output = self._model(**encoded)
        logits = self.classification_head(output[:, 0])
        loss = self._loss_function(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], *args, **kwargs) -> None:
        encoded = self.parse_batch(batch)
        output = self._model(**encoded)
        logits = self.classification_head(output[:, 0])
        for validation_metric in self._evaluation_metrics:
            metric = validation_metric(logits, batch["label"])
            self.log(validation_metric.__class__.__name__, metric)

    def test_step(self, batch: dict[str, Any], *args, **kwargs) -> None:
        self.validation_step(batch, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.AdamW(self.parameters(), lr=5e-5)
        sched = get_constant_schedule_with_warmup(opt, 1000)
        return [opt], [{"scheduler": sched, "interval": "step"}]
