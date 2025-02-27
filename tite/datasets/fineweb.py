from torch.utils.data import DataLoader, Dataset

from .basehfdatamodule import BaseHFDataModule
from .collator import TransformationCollator


class DummyValDataset(Dataset):

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index) -> str:
        return ""


class FineWebDataModule(BaseHFDataModule):
    def __init__(
        self,
        collator: TransformationCollator,
        path: str = "HuggingFaceFW/fineweb",
        name: str = "default",
        data_dir: str | None = None,
        data_files: dict[str, str] | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        num_workers: int = 0,
        streaming: bool = True,
    ) -> None:
        super().__init__(path, collator, name, data_dir, data_files, batch_size, seed, num_workers, streaming)

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        return DataLoader(DummyValDataset())
