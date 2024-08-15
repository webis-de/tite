import random
from typing import Any, Literal, Mapping

import torch
from datasets import load_dataset
from lightning import LightningDataModule

# from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader as StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from ..transformations import StringTransformation
from .commons import seed_or_none


class CollateTokenizerOutput:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int | None, add_special_tokens: bool) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._add_special_tokens = add_special_tokens

    # def __call__(self, batch: list[dict]) -> dict:
    #     agg = {
    #         "teacher_input_ids": [],
    #         "teacher_attention_mask": [],
    #         "student_input_ids": [],
    #         "student_attention_mask": [],
    #         "label": [],
    #     }
    #     for item in batch:
    #         for key, value in item.items():
    #             if key in agg:
    #                 agg[key].append(value)
    #     pad_ids = {"input_ids": self._tokenizer.pad_token_id, "attention_mask": 0, "label": None}
    #     out = {}
    #     for key, values in agg.items():
    #         if not values:
    #             continue
    #         pad_id = pad_ids[key.replace("teacher_", "").replace("student_", "")]
    #         if pad_id is None:
    #             out[key] = torch.tensor(values)
    #         else:
    #             out[key] = torch.nn.utils.rnn.pad_sequence(
    #                 [torch.tensor(value) for value in values], batch_first=True, padding_value=pad_id
    #             )
    #     return out

    def __call__(self, batch: list[dict]) -> dict:
        agg = {"teacher_text_1": [], "student_text_1": [], "teacher_text_2": [], "student_text_2": [], "label": []}
        for item in batch:
            for key, value in item.items():
                if key in agg:
                    agg[key].append(value)
        t1 = agg["teacher_text_1"] + agg["student_text_1"]
        t2 = agg["teacher_text_2"] + agg["student_text_2"] if agg["teacher_text_2"] and agg["student_text_2"] else None
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
        out = {
            "teacher_input_ids": encoded["input_ids"][: len(agg["teacher_text_1"])],
            "teacher_attention_mask": encoded["attention_mask"][: len(agg["teacher_text_1"])],
            "student_input_ids": encoded["input_ids"][len(agg["teacher_text_1"]) :],
            "student_attention_mask": encoded["attention_mask"][len(agg["teacher_text_1"]) :],
        }
        if agg["label"]:
            out["label"] = torch.tensor(agg["label"])
        return out


class BaseHFDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
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
        student_transformations: list[tuple[StringTransformation, float]] | None = None,
        teacher_transformations: list[tuple[StringTransformation, float]] | Literal["student"] | None = None,
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
        self.collate_fn = CollateTokenizerOutput(self._tokenizer, self._max_length, self._add_special_tokens)
        self._student_transformations = student_transformations or []
        self._teacher_transformations = (
            student_transformations if teacher_transformations == "student" else teacher_transformations or []
        )

    def apply_transformations(self, x: dict) -> dict:
        out = {}
        for name, transformations in (
            ("teacher", self._teacher_transformations),
            ("student", self._student_transformations),
        ):
            t1, t2 = x[self._text_keys[0]], x[self._text_keys[1]] if self._text_keys[1] is not None else None
            transformations = transformations or []
            if transformations and t2 is not None:
                raise ValueError("String transformations are not implemented for two texts")
            transformed_t1 = []
            for text in t1:
                for transformation, prob in transformations:
                    if random.random() < prob:
                        text = transformation(text)
                transformed_t1.append(text)
            out[f"{name}_text_1"] = transformed_t1
            if t2 is not None:
                out[f"{name}_text_2"] = t2
        return out

    def tokenize(self, x: dict) -> dict:
        out = {}
        for text_type in ("teacher", "student"):
            encoded = self._tokenizer(
                x[f"{text_type}_text_1"],
                x[f"{text_type}_text_2"] if f"{text_type}_text_2" in x else None,
                truncation=True,
                max_length=self._max_length,
                return_token_type_ids=False,
                add_special_tokens=self._add_special_tokens,
            )
            for key, value in encoded.items():
                out[f"{text_type}_{key}"] = value
        return out

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        assert self._dataset is None, "Dataset is already set up"
        self._dataset = (
            load_dataset(
                path=self.hparams["path"],
                name=self.hparams["name"],
                data_dir=self._data_dir,
                data_files=self._data_files,
                streaming=self._streaming,
            ).map(self.apply_transformations, batched=True)
            # .map(self.tokenize, batched=True)
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
