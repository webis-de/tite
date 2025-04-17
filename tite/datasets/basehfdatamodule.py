import os
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from lightning import LightningDataModule
from torchdata.stateful_dataloader import StatefulDataLoader

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
        self._state_dict = {}

    def setup(self, stage: str) -> None:
        assert self.dataset is None, "Dataset is already set up"
        self.dataset = load_dataset(
            path=self.hparams["path"],
            name=self.hparams["name"],
            **self.data_kwargs,
            streaming=self.streaming,
        )
        # self.dataset = self.dataset.shuffle(seed=self.hparams["seed"])

    def teardown(self, stage: str) -> None:
        self.dataset = None

    def dataloader(self, split: str) -> StatefulDataLoader | None:
        assert self.dataset is not None, "The dataset needs to be set up"
        if split not in self.dataset:
            return None
        loader = StatefulDataLoader(
            self.dataset[split],
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.collator,
            prefetch_factor=16 if self.hparams["num_workers"] > 0 else None,
        )
        if split in self._state_dict:
            loader.load_state_dict(self._state_dict[split])
        return loader

    def train_dataloader(self) -> StatefulDataLoader:
        dataloader = self.dataloader("train")
        assert dataloader is not None, "The train dataloader needs to be set up"
        return dataloader

    def val_dataloader(self) -> StatefulDataLoader | list[StatefulDataLoader]:
        dataloader = self.dataloader("validation")
        return dataloader or []

    def test_dataloader(self) -> StatefulDataLoader | list[StatefulDataLoader]:
        dataloader = self.dataloader("test")
        return dataloader or []

    def state_dict(self) -> dict[str, dict[str, Any]]:
        state_dict = {}
        for split in ("train", "validation", "test"):
            dataloader = self.dataloader(split)
            if dataloader is not None:
                state_dict[split] = dataloader.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, dict[str, Any]]) -> None:
        for split, state in state_dict.items():
            self._state_dict[split] = state
