from typing import Any

import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .commons import seed_or_none


class Collator:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_keys: tuple[str, str | None],
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._add_special_tokens = add_special_tokens
        self._text_keys = text_keys

    def aggregate(self, batch: list[dict]) -> dict:
        agg: dict[str, list] = {key: [] for key in self._text_keys if key is not None}
        agg["label"] = []
        for x in batch:
            for key, value in x.items():
                if key in agg:
                    agg[key].append(value)
        if len(agg["label"]) == 0:
            del agg["label"]
        return agg

    def tokenize(self, agg: dict) -> dict:
        t1 = agg[self._text_keys[0]]
        t2 = None
        if self._text_keys[1] is not None:
            t2 = agg[self._text_keys[1]]
        encoded = self._tokenizer(
            t1,
            t2,
            truncation=True,
            max_length=self._max_length,
            return_token_type_ids=False,
            add_special_tokens=self._add_special_tokens,
            padding=True,
            return_tensors="pt",
        )
        return dict(encoded)

    def __call__(self, batch: list[dict]) -> dict:
        agg = self.aggregate(batch)
        out = self.tokenize(agg)
        if (x := agg.get("label", None)) is not None:
            out["label"] = torch.tensor(x)
        return out


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
        self._data_kwargs: dict[str, Any] = {}
        if data_dir is not None:
            self._data_kwargs["data_dir"] = data_dir
        if data_files is not None:
            self._data_kwargs["data_files"] = data_files
        self._streaming = streaming
        self._dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset | None = None
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
        assert self._dataset is None, "Dataset is already set up"
        self._dataset = load_dataset(
            path=self.hparams["path"],
            name=self.hparams["name"],
            **self._data_kwargs,
            streaming=self._streaming,
        )
        kwargs = {}
        if self._streaming:
            kwargs["buffer_size"] = 1_024
        self._dataset = self._dataset.shuffle(seed=self.hparams["seed"], **kwargs)

    def teardown(self, stage: str) -> None:
        self._dataset = None

    def dataloader(self, split: str) -> DataLoader:
        loader = DataLoader(
            self._dataset[split],
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.collator,
            prefetch_factor=16 if self.hparams["num_workers"] > 0 else None,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        assert self._dataset is not None, "The dataset needs to be set up"
        return self.dataloader("train")

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        assert self._dataset is not None, "The dataset needs to be set up"
        return self.dataloader("validation")

    def test_dataloader(self) -> DataLoader:
        assert self._dataset is not None, "The dataset needs to be set up"
        return self.dataloader("test")

    # def state_dict(self) -> dict[str, dict[str, Any]]:
    #     return {split: loader.state_dict() for split, loader in self._dataloaders.items()}

    # def load_state_dict(self, state_dict: dict[str, dict[str, Any]]) -> None:
    #     for split, state in state_dict.items():
    #         self.dataloader(split).load_state_dict(state)
