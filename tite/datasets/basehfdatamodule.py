from typing import Any, Callable, Literal, Mapping

import torch
from datasets import load_dataset
from lightning import LightningDataModule

# from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader as StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from .commons import seed_or_none


class CollateTokenizerOutput:

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        agg = {"input_ids": [], "attention_mask": [], "label": []}
        for item in batch:
            for key, value in item.items():
                if key in agg:
                    agg[key].append(value)
        pad_ids = {"input_ids": self._tokenizer.pad_token_id, "attention_mask": 0, "label": None}
        out = {}
        for key, values in agg.items():
            if not values:
                continue
            pad_id = pad_ids[key]
            if pad_id is None:
                out[key] = torch.tensor(values)
            else:
                out[key] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(value) for value in values], batch_first=True, padding_value=pad_id
                )
        return out


class BaseHFDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase | None,
        name: str | None = None,
        data_dir: str | None = None,
        data_files: Mapping[str, str] | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        num_workers: int = 0,
        streaming: bool = True,
        max_length: int | None = None,
        text_keys: tuple[str, str | None] = ("text", None),
        add_special_tokens: bool = True,
        collate_fn: Callable[[dict], dict] | None = None,
    ) -> None:
        """
        Args:
            path (str): The hugging face datasets path of the dataset.
            name (str): The hugging face datasets name of the dataset.
            data_dir (str | None, optional): The path to store the dataset at. Defaults to None.
            batch_size (int | None, optional): The number of texts per batch. Defaults to None.
            seed (int | None, optional): The seed to use. If None, pytorch lightning's seed everything seed it used or
                random if that is not set. Defaults to None.
            num_workers (int, optional): The number of workers to employ for the dataloader. Defaults to 0.
            streaming (bool, optional): If set, streams the data on-the-fly instead of downloading it to local storage.
                Defaults to True.
        """
        super().__init__()
        self._data_dir = data_dir
        self._data_files = data_files
        self._streaming = streaming
        self._dataset = None
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
        self._dataloaders: dict[str, StatefulDataLoader] = dict()
        self._max_length = max_length
        self._text_keys = text_keys
        self._add_special_tokens = add_special_tokens
        self._tokenizer = tokenizer
        self.collate_fn = collate_fn if collate_fn is not None else CollateTokenizerOutput(self._tokenizer)

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        assert self._dataset is None, "Dataset is already set up"
        t1, t2 = self._text_keys
        self._dataset = (
            load_dataset(
                path=self.hparams["path"],
                name=self.hparams["name"],
                data_dir=self._data_dir,
                data_files=self._data_files,
                streaming=self._streaming,
            )
            .map(
                lambda x: self._tokenizer(
                    x[t1],
                    x[t2] if t2 is not None else None,
                    truncation=True,
                    max_length=self._max_length,
                    return_token_type_ids=False,
                    add_special_tokens=self._add_special_tokens,
                ),
                batched=True,
            )
            .shuffle(buffer_size=1_024, seed=self.hparams["seed"])
        )

    def teardown(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        self._dataset = None

    def dataloader(self, split: str) -> StatefulDataLoader:
        if (x := self._dataloaders.get(split, None)) is not None:
            return x
        loader = StatefulDataLoader(
            self._dataset[split],
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.collate_fn,
            prefetch_factor=16 if self.hparams["num_workers"] > 0 else None,
        )
        self._dataloaders[split] = loader
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

    def state_dict(self) -> dict[str, dict[str, Any]]:
        return {split: loader.state_dict() for split, loader in self._dataloaders.items()}

    def load_state_dict(self, state_dict: dict[str, dict[str, Any]]) -> None:
        for split, state in state_dict.items():
            self.dataloader(split).load_state_dict(state)
