from typing import NamedTuple

from torch import DoubleTensor, LongTensor
from torch.utils.data import DataLoader, Dataset

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


class DummyValDataset(Dataset):

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index) -> str:
        return ""


class FineWebDataModule(BaseHFDataModule):
    def __init__(self, path: str = "HuggingFaceFW/fineweb", name: str = "CC-MAIN-2024-10", **kwargs) -> None:
        """
        Args:
            name (str, optional): The name of the dump to use (see
                https://huggingface.co/datasets/HuggingFaceFW/fineweb#breakdown-by-dumpcrawl). Defaults to
                "CC-MAIN-2024-10".
        """
        super().__init__(path=path, name=name, **kwargs)

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        return DataLoader(DummyValDataset())
