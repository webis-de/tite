from abc import ABC, abstractmethod
from typing import Any


class Teacher(ABC):
    @abstractmethod
    def map_targets(self, *args, **kwargs):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
