from typing import Literal, NamedTuple
import os

from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch import DoubleTensor, LongTensor


class FWBatch(NamedTuple):
    text: list[str]
    id: list[str]
    dump: list[str]
    url: list[str]
    date: list[str]
    language: list[str]
    language_score: DoubleTensor
    token_count: LongTensor


class FineWebDataModule(LightningDataModule):
    hf_dataset_path = "HuggingFaceFW/fineweb"

    def __init__(
        self,
        name: str = "CC-MAIN-2024-10",
        batch_size: int | None = None,
        seed: int | None = None,
        num_workers: int = 0,
        stream: bool = True,
    ) -> None:
        """
        Args:
            name (str, optional): The name of the dump to use (see
                https://huggingface.co/datasets/HuggingFaceFW/fineweb#breakdown-by-dumpcrawl). Defaults to
                "CC-MAIN-2024-10".
            batch_size (int | None, optional): The number of texts per batch. Defaults to None.
            seed (int | None, optional): The seed to use. If None, pytorch lightning's seed everything seed it used or
                random if that is not set. Defaults to None.
            num_workers (int, optional): The number of workers to employ for the dataloader. Defaults to 0.
            stream (bool, optional): If set, streams the data on-the-fly instead of downloading it to local storage.
                Defaults to True.
        """
        super().__init__()
        seed = (
            int(os.environ.get("PL_GLOBAL_SEED", default=None))
            if seed is None
            else seed
        )
        self.save_hyperparameters(
            {
                "name": name,
                "batch_size": batch_size,
                "seed": seed,
                "num_workers": num_workers,
            }
        )
        self.dataset = None
        self.name = name
        self.stream = stream

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        if stage != "fit":
            raise KeyError("FineWeb only has a training split")
        assert self.dataset is None, "Dataset is already set up"
        self.dataset = (
            load_dataset(
                FineWebDataModule.hf_dataset_path,
                name=self.name,
                split="train",
                streaming=self.stream,
            )
            .with_format("torch")
            .shuffle(buffer_size=10000, seed=self.hparams["seed"])
        )

        self.state_dict = self.dataset.state_dict
        self.load_state_dict = self.dataset.load_state_dict

    def teardown(self) -> None:
        self.dataset = None

    def train_dataloader(self) -> DataLoader:
        assert self.dataset is not None, "Dataset needs to be set up"
        # Note that we don't shuffle here -- we do that in setup() since IterableDatasets can't be shuffled otherwise
        return DataLoader(
            self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )
