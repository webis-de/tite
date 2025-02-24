from abc import ABC, abstractmethod

import torch


class Teacher(ABC):
    @abstractmethod
    def map_targets(self, *args, **kwargs) -> torch.Tensor:
        pass

    def __call__(self, *args: ..., **kwargs: ...) -> ...:
        return self.map_targets(*args, **kwargs)


class IdentityTeacher(Teacher):
    def __init__(self) -> None:
        super().__init__()

    def map_targets(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x
