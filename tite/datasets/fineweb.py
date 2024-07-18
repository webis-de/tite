from typing import NamedTuple

from torch import DoubleTensor, LongTensor

from .basehfdatamodule import BaseHFDataModule


class FWBatch(NamedTuple):
    text: list[str]
    id: list[str]
    dump: list[str]
    url: list[str]
    date: list[str]
    language: list[str]
    language_score: DoubleTensor
    token_count: LongTensor


class FineWebDataModule(BaseHFDataModule):
    def __init__(self, name: str = "CC-MAIN-2024-10", **kwargs) -> None:
        """
        Args:
            name (str, optional): The name of the dump to use (see
                https://huggingface.co/datasets/HuggingFaceFW/fineweb#breakdown-by-dumpcrawl). Defaults to
                "CC-MAIN-2024-10".
        """
        super().__init__(path="HuggingFaceFW/fineweb", name=name, **kwargs)
