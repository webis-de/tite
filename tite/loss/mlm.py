import torch
from torch.nn import Module


class MLMCrossEntropy(Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits, labels):
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        return torch.nn.functional.cross_entropy(logits, labels)


class OrigMLMCrossEntropy(Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits, labels):
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        return torch.nn.functional.cross_entropy(logits, labels)


class BOWBinaryCrossEntropy(Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits, labels):
        logits = logits.view(-1)
        labels = labels.view(-1)
        mask = labels == -100
        logits = logits[~mask]
        labels = labels[~mask]
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
