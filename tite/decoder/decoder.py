from abc import ABC, abstractmethod
from typing import Callable

import torch


class Decoder(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    __call__: Callable[..., torch.Tensor]


class Identity(Decoder):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x
