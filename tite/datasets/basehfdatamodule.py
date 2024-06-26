import os
from typing import Any, Callable, Literal

from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


def _seed_or_none(seed: int | None = None) -> int | None:
    if seed is not None:
        return seed
    seed = os.environ.get("PL_GLOBAL_SEED", default=None)
    return int(seed) if seed is not None else None


class BaseHFDataModule(LightningDataModule):

    def __init__(
        self,
        path: str,
        name: str,
        data_dir: str | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        num_workers: int = 0,
        streaming: bool = True,
        collate_fn: Callable[[list], Any] | None = None,
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
        self._streaming = streaming
        self._dataset = None
        self._collate_fn = collate_fn
        seed = _seed_or_none(seed)
        self.save_hyperparameters(
            {
                "path": path,
                "name": name,
                "batch_size": batch_size,
                "seed": seed,
                "num_workers": num_workers,
            }
        )

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        assert self._dataset is None, "Dataset is already set up"
        self._dataset = (
            load_dataset(
                path=self.hparams["path"],
                name=self.hparams["name"],
                data_dir=self._data_dir,
                streaming=self._streaming,
            )
            .with_format(type="torch")
            .shuffle(buffer_size=10_000, seed=self.hparams["seed"])
        )
        # Maybe for the future: implement state_dict and load_state_dict
        # self.state_dict = self._dataset.state_dict
        # self.load_state_dict = self._dataset.load_state_dict

    def teardown(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        self._dataset = None

    def train_dataloader(self) -> DataLoader:
        assert self._dataset is not None, "The dataset needs to be set up"
        if "train" not in self._dataset:
            return super().val_dataloader()  # Act as if we did not implement this
        # Note that we don't shuffle here -- we do that in setup() since IterableDatasets can't be shuffled otherwise
        return DataLoader(
            self._dataset["train"],
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        assert self._dataset is not None, "The dataset needs to be set up"
        valsplits = [
            DataLoader(
                split,
                batch_size=self.hparams["batch_size"],
                num_workers=self.hparams["num_workers"],
                collate_fn=self._collate_fn,
            )
            for name, split in self._dataset.items()
            if "validation" in name
        ]
        if not valsplits:
            return super().val_dataloader()  # Act as if we did not implement this
        return valsplits[0] if len(valsplits) == 1 else valsplits

    def test_dataloader(self) -> DataLoader:
        assert self._dataset is not None, "The dataset needs to be set up"
        if "test" not in self._dataset:
            return super().val_dataloader()  # Act as if we did not implement this
        return DataLoader(
            self._dataset["train"],
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self._collate_fn,
        )
