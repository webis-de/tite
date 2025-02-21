from abc import ABC, abstractmethod
from typing import Callable

import torch


class Decoder(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    __call__: Callable[..., None]


class Foo(Decoder):

    def forward(self, *args, **kwargs):
        pass


class Bar:
    pass


def test(decoder: Callable[..., None]):
    pass


test(Foo())
test(Bar())
