from abc import ABC, abstractmethod

import torch


class Transformation(ABC):

    def __init__(self, transformation_prob: float = 1.0) -> None:
        super().__init__()
        self.transformation_prob = transformation_prob

    @abstractmethod
    def transform(self, *args, **kwargs) -> ...: ...

    def __call__(self, *args: ..., **kwargs: ...) -> ...:
        return self.transform(*args, **kwargs)

    def transformation_mask(self, batch_size: int) -> torch.Tensor:
        if self.transformation_prob == 1.0:
            return torch.ones(batch_size, dtype=torch.bool)
        if self.transformation_prob == 0.0:
            return torch.zeros(batch_size, dtype=torch.bool)
        return torch.rand(batch_size) < self.transformation_prob
