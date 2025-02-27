from abc import ABC, abstractmethod
from typing import Callable

import torch

from .loss import LossFunction


class LanguageModelingLoss(LossFunction, ABC):

    @abstractmethod
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...

    __call__: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class MLMCrossEntropyLoss(LanguageModelingLoss):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        return torch.nn.functional.cross_entropy(logits, labels)


class MAECrossEntropyLoss(MLMCrossEntropyLoss):
    pass


class BOWBinaryCrossEntropyLoss(LanguageModelingLoss):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        labels = labels.view(-1)
        mask = labels == -100
        logits = logits[~mask]
        labels = labels[~mask]
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
