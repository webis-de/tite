from typing import Literal

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from ..transformation import StringTransformation, TokenTransformation


class Collator:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_keys: tuple[str, str | None],
        max_length: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_keys = text_keys

    def aggregate(self, batch: list[dict]) -> dict:
        agg: dict[str, list] = {key: [] for key in self.text_keys if key is not None}
        agg["label"] = []
        for x in batch:
            for key, value in x.items():
                if key in agg:
                    agg[key].append(value)
        if len(agg["label"]) == 0:
            del agg["label"]
        return agg

    def tokenize(self, agg: dict) -> BatchEncoding:
        t1 = agg[self.text_keys[0]]
        t2 = None
        if self.text_keys[1] is not None:
            t2 = agg[self.text_keys[1]]
        encoded = self.tokenizer(
            t1,
            t2,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        return encoded

    def __call__(self, batch: list[dict]) -> BatchEncoding:
        agg = self.aggregate(batch)
        out = self.tokenize(agg)
        if (x := agg.get("label", None)) is not None:
            out["label"] = torch.tensor(x)
        return out


class TransformationCollator(Collator):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_keys: tuple[str, str | None],
        string_transformations: dict[str, list[StringTransformation] | None] | None = None,
        token_transformations: dict[str, list[TokenTransformation] | None] | None = None,
        max_length: int | None = None,
    ) -> None:
        if text_keys[1] is not None:
            raise ValueError("Text pairs are not supported")
        super().__init__(tokenizer, text_keys, max_length)
        string_transformations = string_transformations or {}
        self.string_transformations = {}
        for heads, transformations in string_transformations.items():
            self.string_transformations[tuple(heads.split(","))] = transformations or []
        token_transformations = token_transformations or {}
        self.token_transformations = {}
        for heads, transformations in token_transformations.items():
            self.token_transformations[tuple(heads.split(","))] = transformations or []

    def apply_string_transformations(
        self, agg: dict, string_transformations: list[StringTransformation]
    ) -> tuple[tuple[str], dict]:
        text_key = self.text_keys[0]
        texts = agg[text_key]
        auxiliary_data = {}
        transformed_idcs_and_texts = [(idx, text) for idx, text in enumerate(texts)]
        for transformation in string_transformations:
            transformed_idcs_and_texts, transform_auxiliary_data = transformation(transformed_idcs_and_texts)
            auxiliary_data = {**auxiliary_data, **transform_auxiliary_data}
        batch_idcs, transformed_texts = zip(*transformed_idcs_and_texts)
        auxiliary_data["batch_idcs"] = batch_idcs
        return transformed_texts, auxiliary_data

    def apply_token_transformations(
        self, encoding: BatchEncoding, token_transformations: list[TokenTransformation]
    ) -> tuple[BatchEncoding, dict]:
        auxiliary_data = {}
        transformed_encoding = encoding
        for transformation in token_transformations:
            transformed_encoding, transform_auxiliary_data = transformation(transformed_encoding)
            auxiliary_data = {**auxiliary_data, **transform_auxiliary_data}
        return transformed_encoding, auxiliary_data

    def tokenize(self, agg: dict) -> tuple[BatchEncoding, BatchEncoding, dict]:
        encoding = super().tokenize(agg)
        transformed_texts = {}
        string_auxiliary_data = {}
        transformed_encodings = {}
        token_auxiliary_data = {}
        for head, string_transformations in self.string_transformations.items():
            transformed_texts[head], string_auxiliary_data[head] = self.apply_string_transformations(
                agg, string_transformations
            )
            transformed_encodings[head] = super().tokenize({self.text_keys[0]: transformed_texts[head]})
        for head, token_transformations in self.token_transformations.items():
            transformed_encodings[head], token_auxiliary_data[head] = self.apply_token_transformations(
                transformed_encodings[head], token_transformations
            )
        return encoding, transformed_encodings, {**string_auxiliary_data, **token_auxiliary_data}
