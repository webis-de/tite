import os
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .collator import Collator


def seed_or_none(seed: int | None = None) -> int | None:
    if seed is not None:
        return seed
    seed = os.environ.get("PL_GLOBAL_SEED", default=None)
    return int(seed) if seed is not None else None


class BaseHFDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        collator: Collator,
        name: str = "default",
        data_dir: str | None = None,
        data_files: dict[str, str] | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        num_workers: int = 0,
        streaming: bool = True,
    ) -> None:
        super().__init__()
        self.data_kwargs: dict[str, Any] = {}
        if data_dir is not None:
            self.data_kwargs["data_dir"] = data_dir
        if data_files is not None:
            self.data_kwargs["data_files"] = data_files
        self.streaming = streaming
        self.dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset | None = None
        self.collator = collator
        seed = seed_or_none(seed)
        self.save_hyperparameters(
            {
                "path": path,
                "name": name,
                "batch_size": batch_size,
                "seed": seed,
                "num_workers": num_workers,
            }
        )

    def setup(self, stage: str) -> None:
        assert self.dataset is None, "Dataset is already set up"
        self.dataset = load_dataset(
            path=self.hparams["path"],
            name=self.hparams["name"],
            **self.data_kwargs,
            streaming=self.streaming,
        )
        kwargs = {}
        if self.streaming:
            kwargs["buffer_size"] = 1_024
        self.dataset = self.dataset.shuffle(seed=self.hparams["seed"], **kwargs)

    def teardown(self, stage: str) -> None:
        self.dataset = None

    def dataloader(self, split: str) -> DataLoader:
        loader = DataLoader(
            self.dataset[split],
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.collator,
            prefetch_factor=16 if self.hparams["num_workers"] > 0 else None,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        assert self.dataset is not None, "The dataset needs to be set up"
        return self.dataloader("train")

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        assert self.dataset is not None, "The dataset needs to be set up"
        return self.dataloader("validation")

    def test_dataloader(self) -> DataLoader:
        assert self.dataset is not None, "The dataset needs to be set up"
        return self.dataloader("test")

    # def state_dict(self) -> dict[str, dict[str, Any]]:
    #     return {split: loader.state_dict() for split, loader in self.dataloaders.items()}

    # def load_state_dict(self, state_dict: dict[str, dict[str, Any]]) -> None:
    #     for split, state in state_dict.items():
    #         self.dataloader(split).load_state_dict(state)
