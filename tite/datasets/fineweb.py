from typing import Any, Literal

from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from ..transformations import CopyStudentTransformation, StringTransformation, TokenTransformation
from .basehfdatamodule import BaseHFDataModule, Collator


class DummyValDataset(Dataset):

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index) -> str:
        return ""


class TransformationCollator(Collator):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_keys: tuple[str, str | None],
        add_special_tokens: bool = True,
        max_length: int | None = None,
        student_string_transformations: list[StringTransformation] | None = None,
        teacher_string_transformations: list[StringTransformation] | Literal["student", False] | None = None,
        student_token_transformations: list[TokenTransformation] | None = None,
        teacher_token_transformations: list[TokenTransformation] | Literal["student", False] | None = None,
    ) -> None:
        if text_keys[1] is not None:
            raise ValueError("Text pairs are not supported")
        super().__init__(tokenizer, text_keys, max_length, add_special_tokens)
        self._student_string_transformations = student_string_transformations or []
        if teacher_string_transformations == "student":
            self._teacher_string_transformations: list[StringTransformation] | None = (
                self._student_string_transformations
            )
        elif teacher_string_transformations is False:
            self._teacher_string_transformations = None
        else:
            self._teacher_string_transformations = teacher_string_transformations or []
        self._student_token_transformations = student_token_transformations or []
        if teacher_token_transformations == "student":
            self._teacher_token_transformations: list[TokenTransformation] | None = self._student_token_transformations
        elif teacher_token_transformations is False:
            self._teacher_token_transformations = None
        else:
            self._teacher_token_transformations = teacher_token_transformations or []

    def apply_string_transformations(self, agg: dict) -> tuple[dict, dict]:
        transformed: dict[str, list] = {}
        text_key = self._text_keys[0]
        aux: dict[str, Any] = {}
        for name, transformations in (
            ("student", self._student_string_transformations),
            ("teacher", self._teacher_string_transformations),
        ):
            texts = agg[text_key]
            if transformations is None:
                continue
            if transformations and isinstance(transformations[0], CopyStudentTransformation):
                for key, item in list(transformed.items()):
                    if key.startswith("student_"):
                        transformed[key.replace("student_", "teacher_")] = item
            else:
                transformed_idcs_and_texts = [(idx, text) for idx, text in enumerate(texts)]
                for transformation in transformations:
                    transformed_idcs_and_texts, transform_aux = transformation(transformed_idcs_and_texts)
                    aux = {**aux, **transform_aux}
                batch_idcs, transformed_texts = zip(*transformed_idcs_and_texts)
                transformed[f"{name}_{text_key}"] = list(transformed_texts)
                aux[f"{name}_batch_idcs"] = batch_idcs
        agg[text_key] = transformed[f"student_{text_key}"] + transformed.get(f"teacher_{text_key}", [])
        return agg, aux

    def apply_token_transformations(self, encoded: dict) -> tuple[dict, dict]:
        aux: dict[str, Any] = {}
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if self._teacher_string_transformations is None:
            student_input = {"input_ids": input_ids, "attention_mask": attention_mask}
            teacher_input = None
        else:
            student_input = {
                "input_ids": input_ids[: len(input_ids) // 2],
                "attention_mask": attention_mask[: len(input_ids) // 2],
            }
            teacher_input = {
                "input_ids": input_ids[len(input_ids) // 2 :],
                "attention_mask": attention_mask[len(input_ids) // 2 :],
            }
        for transformation in self._student_token_transformations:
            student_input, transform_aux = transformation(**student_input)
            aux = {**aux, **transform_aux}
        out = {"student_input": student_input}
        if teacher_input is not None and self._teacher_token_transformations is not None:
            for transformation in self._teacher_token_transformations:
                teacher_input, transform_aux = transformation(**teacher_input)
                aux = {**aux, **transform_aux}
            out["teacher_input"] = teacher_input
        return out, aux

    def tokenize(self, agg: dict) -> dict:
        agg, string_aux = self.apply_string_transformations(agg)
        encoded = super().tokenize(agg)
        encoded, token_aux = self.apply_token_transformations(encoded)
        return {**encoded, **string_aux, **token_aux}


class FineWebDataModule(BaseHFDataModule):
    def __init__(
        self,
        collator: TransformationCollator,
        path: str = "HuggingFaceFW/fineweb",
        name: str = "default",
        data_dir: str | None = None,
        data_files: dict[str, str] | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        num_workers: int = 0,
        streaming: bool = True,
    ) -> None:
        super().__init__(path, collator, name, data_dir, data_files, batch_size, seed, num_workers, streaming)

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        return DataLoader(DummyValDataset())
