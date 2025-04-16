from pathlib import Path

from tite.datasets.basehfdatamodule import BaseHFDataModule
from tite.datasets.collator import Collator
from tite.model.tokenizer import TiteTokenizer


def test_resume(tokenizer: TiteTokenizer) -> None:
    data_dir = Path(__file__).parent / "data"
    datamodule = BaseHFDataModule(
        path="csv",
        collator=Collator(tokenizer, text_keys=("text", None), max_length=8),
        batch_size=2,
        data_dir=None,
        seed=42,
        data_files={"train": str(data_dir / "dummy-text.csv")},
        num_workers=0,
        streaming=True,
    )
    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader()

    iterator = iter(dataloader)
    first_sample = next(iterator)
    state_dict = datamodule.state_dict()
    second_sample = next(iterator)
    assert (first_sample["input_ids"] != second_sample["input_ids"]).any()

    datamodule = BaseHFDataModule(
        path="csv",
        collator=Collator(tokenizer, text_keys=("text", None), max_length=8),
        batch_size=2,
        data_dir=None,
        seed=42,
        data_files={"train": str(data_dir / "dummy-text.csv")},
        num_workers=0,
        streaming=True,
    )
    datamodule.setup(stage="fit")
    datamodule.load_state_dict(state_dict)
    dataloader = datamodule.train_dataloader()

    new_second_sample = next(iter(dataloader))

    assert (second_sample["input_ids"] == new_second_sample["input_ids"]).all()
