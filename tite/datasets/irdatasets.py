from collections import defaultdict
from typing import Literal

import ir_datasets
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from ..model.tokenizer import TiteTokenizer
from .commons import seed_or_none

SplitType = Literal["qrels", "triples", "scoreddocs"]
SplitDescriptor = tuple[str, SplitType]


class Collator:

    def __init__(
        self,
        tokenizer: TiteTokenizer,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

    def aggregate(self, batch: list[dict]) -> dict:
        agg = defaultdict(list)
        for x in batch:
            for key, value in x.items():
                agg[key].append(value)
        return dict(agg)

    def tokenize(self, agg: dict) -> dict:
        for key, value in list(agg.items()):
            if key.endswith("id"):
                continue
            agg[f"encoded_{key}"] = self.tokenizer(
                value,
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                add_special_tokens=self.add_special_tokens,
                padding=True,
                return_tensors="pt",
            )
        return agg

    def __call__(self, batch: list[dict]) -> dict:
        agg = self.aggregate(batch)
        out = self.tokenize(agg)
        return out


class IRDataset:

    def __init__(self, dataset_id: str) -> None:
        self.dataset = ir_datasets.load(dataset_id)
        self.doc_store = self.dataset.docs_store()
        self.queries = pd.DataFrame(self.dataset.queries_iter()).set_index("query_id")["text"]


class TripleDataset(IRDataset, IterableDataset):

    def __init__(self, dataset_id: str) -> None:
        super().__init__(dataset_id)

    def __iter__(self):
        for docpair in self.dataset.docpairs_iter():
            query = self.queries.loc[docpair.query_id]
            pos_doc = self.doc_store.get(docpair.doc_id_a).default_text()
            neg_doc = self.doc_store.get(docpair.doc_id_b).default_text()
            yield {
                "query_id": docpair.query_id,
                "pos_doc_id": docpair.doc_id_a,
                "neg_doc_id": docpair.doc_id_b,
                "query": query,
                "pos_doc": pos_doc,
                "neg_doc": neg_doc,
            }


class ScoredDocsDataset(IRDataset, IterableDataset):
    def __init__(self, dataset_id: str) -> None:
        super().__init__(dataset_id)

    def __iter__(self):
        for scoreddoc in iter(self.dataset.scoreddocs_iter()):
            query = self.queries.loc[scoreddoc.query_id]
            doc = self.doc_store.get(scoreddoc.doc_id).default_text()
            yield {
                "query_id": scoreddoc.query_id,
                "doc_id": scoreddoc.doc_id,
                "query": query,
                "doc": doc,
                "dataset_id": self.dataset.dataset_id(),
            }


SPLIT_TYPE_TO_DATASET = {
    "triples": TripleDataset,
    "scoreddocs": ScoredDocsDataset,
}


class IRDatasetsDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: TiteTokenizer,
        add_special_tokens: bool = True,
        trainset: SplitDescriptor | None = None,
        valset: SplitDescriptor | None = None,
        testset: SplitDescriptor | None = None,
        batch_size: int = 32,
        inference_batch_size: int = 256,
        seed: int | None = None,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        seed = seed_or_none(seed)
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.collator = Collator(tokenizer, max_length=max_length, add_special_tokens=add_special_tokens)

    def train_dataloader(self) -> DataLoader:
        if self.trainset is None:
            raise ValueError("No training set provided")
        return DataLoader(
            SPLIT_TYPE_TO_DATASET[self.trainset[1]](self.trainset[0]),
            batch_size=self.batch_size,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        if self.valset is None:
            raise ValueError("No validation set provided")
        return DataLoader(
            SPLIT_TYPE_TO_DATASET[self.valset[1]](self.valset[0]),
            batch_size=self.inference_batch_size,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        if self.testset is None:
            raise ValueError("No test set provided")
        return DataLoader(
            SPLIT_TYPE_TO_DATASET[self.testset[1]](self.testset[0]),
            batch_size=self.inference_batch_size,
            collate_fn=self.collator,
        )
