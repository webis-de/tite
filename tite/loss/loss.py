from abc import ABC, abstractmethod
from typing import Callable

import torch


class LossFunction(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    __call__: Callable[..., torch.Tensor]
