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
        encoder_string_transformations: list[StringTransformation] | None,
        decoder_string_transformations: list[list[StringTransformation] | Literal["encoder"] | None],
        encoder_token_transformations: list[TokenTransformation] | None,
        decoder_token_transformations: list[list[TokenTransformation] | Literal["encoder"] | None],
        max_length: int | None = None,
    ) -> None:
        if text_keys[1] is not None:
            raise ValueError("Text pairs are not supported")
        super().__init__(tokenizer, text_keys, max_length)
        self.encoder_string_transformations = encoder_string_transformations or []
        self.decoder_string_transformations = [
            self.encoder_string_transformations if string_transformations == "encoder" else string_transformations or []
            for string_transformations in decoder_string_transformations
        ]
        self.encoder_token_transformations = encoder_token_transformations or []
        self.decoder_token_transformations = [
            self.encoder_token_transformations if token_transformations == "encoder" else token_transformations or []
            for token_transformations in decoder_token_transformations
        ]
        if len(self.decoder_string_transformations) != len(self.decoder_token_transformations):
            raise ValueError("Number of decoder string and token transformations must match")

    def apply_string_transformations(self, agg: dict) -> tuple[list[list[str]], list[dict]]:
        all_transformed_texts: list[list[str]] = []
        all_auxiliary_data: list[dict] = []
        text_key = self.text_keys[0]
        texts = agg[text_key]
        for transformations in [self.encoder_string_transformations, *self.decoder_string_transformations]:
            auxiliary_data = {}
            transformed_idcs_and_texts = [(idx, text) for idx, text in enumerate(texts)]
            for transformation in transformations:
                transformed_idcs_and_texts, transform_auxiliary_data = transformation(transformed_idcs_and_texts)
                auxiliary_data = {**auxiliary_data, **transform_auxiliary_data}
            batch_idcs, transformed_texts = zip(*transformed_idcs_and_texts)
            all_transformed_texts.append(list(transformed_texts))
            auxiliary_data["batch_idcs"] = batch_idcs
            all_auxiliary_data.append(auxiliary_data)
        return all_transformed_texts, all_auxiliary_data

    def apply_token_transformations(self, encodings: list[BatchEncoding]) -> tuple[list[BatchEncoding], list[dict]]:
        all_transformed_encodings = []
        all_auxiliary_data: list[dict] = []
        all_transformations = [self.encoder_token_transformations] + self.decoder_token_transformations
        assert len(encodings) == len(all_transformations)
        for encoding, transformations in zip(encodings, all_transformations):
            auxiliary_data = {}
            transformed_encoding = encoding
            for transformation in transformations:
                transformed_encoding, transform_auxiliary_data = transformation(encoding)
                auxiliary_data = {**auxiliary_data, **transform_auxiliary_data}
            all_transformed_encodings.append(transformed_encoding)
            all_auxiliary_data.append(auxiliary_data)
        return all_transformed_encodings, all_auxiliary_data

    def tokenize(self, agg: dict) -> list[tuple[BatchEncoding, dict]]:
        transformed_texts, string_auxiliary_data = self.apply_string_transformations(agg)
        encodings = []
        for texts in transformed_texts:
            encodings.append(super().tokenize({self.text_keys[0]: texts}))
        encodings, token_auxiliary_data = self.apply_token_transformations(encodings)
        out = []
        for encoding, string_data, token_data in zip(encodings, string_auxiliary_data, token_auxiliary_data):
            out.append((encoding, {**string_data, **token_data}))
        return out
