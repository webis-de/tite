from typing import Any, Callable, Iterable, Literal

import ir_datasets
import pandas as pd
import torch
from lightning import LightningDataModule
from more_itertools import batched as mit_batched
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase

from .commons import seed_or_none

SplitType = Literal["qrels", "triples"]
SplitDescriptor = tuple[str, SplitType]


def unbatch(batches: Iterable[Iterable]) -> Iterable:
    for b in batches:
        yield from b


class TokenizeMeSenpai:
    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


class ApplyTokenizerTransform:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs) -> None:
        self._tokenizer = tokenizer
        self._kwargs = kwargs

    def __apply_tokenizer(self, batch: list[str]) -> list[dict]:
        tokenized = self._tokenizer(batch, **self._kwargs)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        return [{"input_ids": i, "attention_mask": a} for i, a in zip(input_ids, attention_mask)]

    def __apply(self, val: Iterable[Any]) -> Iterable[Any]:
        val = list(val)
        if isinstance(val[0], TokenizeMeSenpai):
            return self.__apply_tokenizer([str(v) for v in val])
        return val

    def __call__(self, batch: list[tuple[Any, ...]]) -> Iterable[tuple[dict, dict, dict]]:
        tuples = zip(*batch)
        return list(zip(*map(self.__apply, tuples)))


class CollateTokenizerOutput:

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def __collate_tokens(self, batch: list[dict]) -> dict:
        agg = {"input_ids": [], "attention_mask": []}
        for item in batch:
            for key, value in item.items():
                if key in agg:
                    agg[key].append(value)
        pad_ids = {"input_ids": self._tokenizer.pad_token_id, "attention_mask": 0}
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

    def __collate(self, vals: Iterable[Any]) -> Any:
        vals = list(vals)
        if isinstance(vals[0], dict) and "input_ids" in vals[0] and "attention_mask" in vals[0]:
            return self.__collate_tokens(vals)
        return vals

    def __call__(self, batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
        tuples = zip(*batch)
        return tuple(map(self.__collate, tuples))


class IRDataset(IterableDataset):
    def __init__(self, datasetname: str, type: SplitType) -> None:
        super().__init__()
        self._transforms = []
        self._dataset = ir_datasets.load(datasetname)
        if type == "triples":
            self._iter = self._dataset.docpairs_iter()
            self.map(self.__fetch_triple_text, batched=True)
            self._count = self._dataset.docpairs_count()
        else:
            self._iter = self._dataset.qrels_iter()
            self.map(self.__fetch_qrel_text, batched=True)
            self._count = self._dataset.qrels_count()
        self._docstore = self._dataset.docs_store()
        self._qstore = pd.DataFrame(self._dataset.queries_iter()).set_index("query_id")

    def __get_many_queries(self, query_ids: Iterable[str]) -> Any:
        return self._qstore.loc[list(query_ids)].itertuples()

    def __fetch_triple_text(self, triple: list[tuple[str, str, str]]) -> list[tuple[str, str, str, str, str, str]]:
        qids, posdids, negdids = zip(*triple)
        qs = (q.text for q in self.__get_many_queries(qids))
        posdocs = (d.text for _, d in self._docstore.get_many(posdids).items())
        negdocs = (d.text for _, d in self._docstore.get_many(negdids).items())
        return [
            (q, TokenizeMeSenpai(qt), p, TokenizeMeSenpai(pt), n, TokenizeMeSenpai(nt))
            for q, qt, p, pt, n, nt in zip(qids, qs, posdids, posdocs, negdids, negdocs)
        ]

    def __fetch_qrel_text(self, triple: list[tuple[str, str, int, str]]) -> list[tuple[str, str, str, str, int, str]]:
        qids, dids, rels, unused = zip(*triple)
        qs = (q.text for q in self.__get_many_queries(qids))
        ds = (d.text for _, d in self._docstore.get_many(dids).items())
        return [
            (q, TokenizeMeSenpai(qt), p, TokenizeMeSenpai(pt), r, u)
            for q, qt, p, pt, r, u in zip(qids, qs, dids, ds, rels, unused)
        ]

    def map(self, fn: Callable, batched: bool = False, batchsize: int = 1000) -> "IRDataset":
        if not batched:
            self._iter = map(fn, self._iter)
        else:
            self._iter = unbatch(map(fn, mit_batched(self._iter, batchsize)))
        return self

    def shuffle(self, buffer_size: int, seed: int, **kwargs) -> "IRDataset":
        # raise NotImplementedError
        return self

    def __len__(self):
        return self._count

    def __iter__(self):
        return self._iter


class IRDatasetsDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        trainset: SplitDescriptor | None = None,
        valset: SplitDescriptor | None = None,
        testset: SplitDescriptor | None = None,
        data_dir: str | None = None,
        data_files: Callable[[str], str] | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        num_workers: int = 0,
        streaming: bool = True,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        collate_fn: Callable[[dict], dict] | None = None,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self._data_dir = data_dir
        self._data_files = data_files
        self._streaming = streaming
        self._tokenizer = tokenizer
        seed = seed_or_none(seed)
        self.save_hyperparameters(
            {
                "path": path,
                "trainset": trainset,
                "valset": valset,
                "testset": testset,
                "batch_size": batch_size,
                "seed": seed,
                "num_workers": num_workers,
            }
        )
        self._dataloaders: dict[str, DataLoader] = dict()
        self.collate_fn = collate_fn if collate_fn is not None else CollateTokenizerOutput(self._tokenizer)
        self._transform = (
            transform
            if transform is not None
            else ApplyTokenizerTransform(self._tokenizer, max_length=max_length, add_special_tokens=add_special_tokens)
        )

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        pass

    def __dataloader(self, split: SplitDescriptor) -> DataLoader:
        if (x := self._dataloaders.get(split, None)) is not None:
            return x
        path = self.hparams["path"]
        splitname, splittype = split
        dataset = (
            IRDataset(f"{path}/{splitname}", splittype)
            .map(
                self._transform,
                batched=True,
            )
            .shuffle(buffer_size=1_024, seed=self.hparams["seed"])
        )
        loader = DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.collate_fn,
            prefetch_factor=16 if self.hparams["num_workers"] > 0 else None,
        )
        self._dataloaders[split] = loader
        return loader

    def teardown(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(self.hparams["trainset"])

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        return self.__dataloader(self.hparams["valset"])

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader(self.hparams["testset"])

    def state_dict(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, dict[str, Any]]) -> None:
        raise NotImplementedError
