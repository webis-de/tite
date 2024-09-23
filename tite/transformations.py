import random
from typing import Any

import nltk
import torch
from nltk import PunktSentenceTokenizer
from torch import LongTensor, Tensor
from torch.nn import Module


class Transformation(Module):
    def __init__(self, transformation_prob: float = 1.0) -> None:
        super().__init__()
        self.transformation_prob = transformation_prob

    def transformation_mask(self, batch_size: int) -> Tensor:
        return torch.rand(batch_size) < self.transformation_prob

    def forward(self, *args: Any, **kwds: Any) -> tuple[dict, dict]:
        raise NotImplementedError


class TokenTransformation(Transformation):
    pass


class MLMMaskTokens(TokenTransformation):
    def __init__(
        self,
        vocab_size: int,
        mask_id: int,
        cls_id: int,
        sep_id: int,
        mask_prob: float = 0.3,
        transformation_prob: float = 1.0,
    ) -> None:
        super().__init__(transformation_prob)
        self._vocab_size = vocab_size
        self._mask_id = mask_id
        self._cls_id = cls_id
        self._sep_id = sep_id
        self._mask_prob = mask_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        mlm_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._mask_prob
        mlm_mask = mlm_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        mlm_mask = mlm_mask.logical_and(transformation_mask[:, None])
        probability_matrix = torch.rand(attention_mask.shape, device=input_ids.device)
        mask_mask = mlm_mask & (probability_matrix < 0.8)
        mask_random = mlm_mask & (probability_matrix >= 0.8) & (probability_matrix < 0.9)
        input_ids = torch.where(mask_mask, self._mask_id, input_ids)
        input_ids = torch.where(
            mask_random, torch.randint(self._vocab_size, input_ids.shape, device=input_ids.device), input_ids
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}, {"mlm_mask": mlm_mask}


class MaskTokens(TokenTransformation):
    def __init__(
        self, mask_id: int, cls_id: int, sep_id: int, mask_prob: float = 0.3, transformation_prob: float = 1.0
    ) -> None:
        super().__init__(transformation_prob)
        self._mask_id = mask_id
        self._cls_id = cls_id
        self._sep_id = sep_id
        self._mask_prob = mask_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        mlm_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._mask_prob
        mlm_mask = mlm_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        mlm_mask = mlm_mask.logical_and(transformation_mask[:, None])
        input_ids = torch.where(mlm_mask, self._mask_id, input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, {}


class ReplaceRandom(TokenTransformation):

    def __init__(
        self,
        vocab_size: int,
        cls_id: int,
        sep_id: int,
        replace_prob: float = 0.05,
        transformation_prob: float = 1.0,
    ):
        super().__init__(transformation_prob)
        self._vocab_size = vocab_size
        self._cls_id = cls_id
        self._sep_id = sep_id
        self._replace_prob = replace_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        random_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._replace_prob
        random_mask = random_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        random_mask = random_mask.logical_and(transformation_mask[:, None])
        input_ids = torch.where(random_mask, torch.randint_like(input_ids, 0, self._vocab_size), input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, {}


class DeleteTokens(TokenTransformation):
    def __init__(
        self, pad_id: int, cls_id: int, sep_id: int, delete_prob: float = 0.3, transformation_prob: float = 1.0
    ):
        super().__init__(transformation_prob)
        self._pad_id = pad_id
        self._cls_id = cls_id
        self._sep_id = sep_id
        self._delete_prob = delete_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        delete_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._delete_prob
        delete_mask = delete_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        delete_mask = delete_mask.logical_and(transformation_mask[:, None])
        num_delete = delete_mask.sum(-1)
        num_tokens = input_ids.shape[1] - num_delete
        new_input_ids = torch.nn.utils.rnn.pad_sequence(
            torch.split(input_ids[~delete_mask], num_tokens.tolist()), batch_first=True, padding_value=self._pad_id
        )
        new_attention_mask = torch.nn.utils.rnn.pad_sequence(
            torch.split(attention_mask[~delete_mask], num_tokens.tolist()), batch_first=True
        )
        return {"input_ids": new_input_ids, "attention_mask": new_attention_mask}, {}


class InsertRandomTokens(TokenTransformation):
    def __init__(self, vocab_size: int, pad_id: int, insert_prob: float = 0.05, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self._pad_id = pad_id
        self._insert_prob = insert_prob
        self._vocab_size = vocab_size

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        batch_size, orig_seq_len = input_ids.shape
        num_non_zero = attention_mask.sum(-1)
        batch_idx, token_idx = attention_mask.nonzero(as_tuple=True)

        insert_mask = torch.rand((batch_size, orig_seq_len), device=input_ids.device) < self._insert_prob
        insert_mask = torch.where(transformation_mask[:, None], insert_mask, torch.zeros_like(insert_mask))
        insert_idx = torch.max(torch.poisson(insert_mask.float()), insert_mask).long().cumsum(-1)
        num_added = insert_idx[:, -1]
        max_num_added = int(num_added.max().cpu())
        new_seq_len = orig_seq_len + max_num_added

        new_token_idx = token_idx + insert_idx[attention_mask.bool()]
        new_input_ids = torch.full((batch_size, new_seq_len), -100, device=input_ids.device)
        new_input_ids[batch_idx, new_token_idx] = input_ids[batch_idx, token_idx]
        # TODO can sample special tokens here; perhaps fix in the future
        new_input_ids = torch.where(
            new_input_ids == -100,
            torch.randint(self._vocab_size, (batch_size, new_seq_len), device=input_ids.device),
            new_input_ids,
        )

        new_lengths = num_non_zero + num_added
        new_attention_mask = (
            torch.arange(new_seq_len, device=input_ids.device)[None].expand(batch_size, -1) < new_lengths[..., None]
        )
        new_input_ids[~new_attention_mask] = self._pad_id
        return {"input_ids": new_input_ids, "attention_mask": new_attention_mask}, {}


class StringTransformation:

    def __init__(self, transformation_prob: float = 1.0) -> None:
        self.transformation_prob = transformation_prob

    def __call__(self, texts: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], dict]:
        transformed_texts = []
        for batch_idx, text in texts:
            if not text:
                transformed_texts.append((batch_idx, text))
                continue
            for transformed_text in self._transform(text):
                transformed_texts.append((batch_idx, transformed_text))
        return transformed_texts, {}

    def _transform(self, text: str) -> list[str]:
        raise NotImplementedError


class CharacterTransformation(StringTransformation):

    def encode(self, text: str) -> Tensor:
        return torch.tensor(bytearray(text.encode("utf-8")))

    def decode(self, encoded: Tensor) -> str:
        return bytes(bytearray(encoded.tolist())).decode("utf-8")


class SentenceTransformation(StringTransformation):

    def __init__(self, transformation_prob: float = 1.0) -> None:
        super().__init__(transformation_prob)
        self._tokenizer: PunktSentenceTokenizer = nltk.load("tokenizers/punkt/english.pickle")
        assert isinstance(self._tokenizer, PunktSentenceTokenizer)

    def split(self, text: str) -> list[str]:
        return self._tokenizer.tokenize(text)


class CharacterDelete(CharacterTransformation):

    def __init__(self, delete_prob: float = 0.3, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self._delete_prob = delete_prob

    # def _transform(self, texts: list[str]) -> list[str]:
    #     encoded = self.encode(text)
    #     delete_mask = torch.rand(encoded.shape) < self._delete_prob
    #     transformed_encoded = encoded[~delete_mask & (encoded <= 128)]
    #     transformed = self.decode(transformed_encoded)
    #     return transforme

    def _transform(self, text: str) -> list[str]:
        transformed_text = "".join(char for char in text if random.random() > self._delete_prob)
        return [transformed_text]


class CharacterInsert(CharacterTransformation):

    def __init__(self, insert_prob: float = 0.05, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self._insert_prob = insert_prob

    # def _transform(self, text: str) -> list[str]:
    #     encoded = self.encode(text)
    #     chars = encoded.unique()
    #     insert_num = int(torch.distributions.Binomial(encoded.shape[0], self._insert_prob).sample().item())
    #     transformed_encoded = torch.zeros(encoded.shape[0] + insert_num, dtype=encoded.dtype)
    #     insert_idcs = torch.randperm(transformed_encoded.shape[0])[:insert_num]
    #     idcs = torch.arange(transformed_encoded.shape[0])
    #     orig_idcs = idcs[(idcs[:, None] != insert_idcs[None]).all(-1)]
    #     transformed_encoded[insert_idcs] = torch.randint(chars.shape[0], (insert_num,))
    #     transformed_encoded[orig_idcs] = encoded
    #     transformed = self.decode(transformed_encoded)
    #     return transformed

    def _transform(self, text: str) -> list[str]:
        transformed_text = "".join(
            char if random.random() > self._insert_prob else char + random.choice(text) for char in text
        )
        return [transformed_text]


class CharacterSwapNeighboring(CharacterTransformation):

    def __init__(self, swap_prob: float = 0.05, transformation_prob: float = 1.0):
        self._swap_prob = swap_prob

    # def _transform(self, text: str) -> list[str]:
    #     encoded = self.encode(text)
    #     swap_mask = torch.rand(encoded.shape[0] - 1) < self._swap_prob
    #     encoded_transformed = encoded.clone()
    #     encoded_transformed[:-1][swap_mask] = encoded[1:][swap_mask]
    #     encoded_transformed[1:][swap_mask] = encoded[:-1][swap_mask]
    #     transformed = self.decode(encoded_transformed)
    #     return transformed

    def _transform(self, text: str) -> list[str]:
        transformed_text = list(text)
        num_swaps = int(torch.distributions.Binomial(len(text), self._swap_prob).sample().item())
        for _ in range(num_swaps):
            i = random.randint(0, len(text) - 2)
            transformed_text[i], transformed_text[i + 1] = transformed_text[i + 1], transformed_text[i]
        return ["".join(transformed_text)]


class SentenceDelete(SentenceTransformation):

    def __init__(self, delete_prob: float = 0.3, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self._delete_prob = delete_prob

    def _transform(self, text: str) -> list[str]:
        if not text:
            return [text]
        sentences = self.split(text)
        keep = [random.random() >= self._delete_prob for _ in range(len(sentences))]
        if not any(keep):
            keep[random.randint(0, len(keep) - 1)] = True
        return [" ".join(sentence for sentence, keep_sentence in zip(sentences, keep) if keep_sentence)]


class SentenceSwapNeighboring(SentenceTransformation):

    def __init__(self, swap_prob: float = 0.05, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self._swap_prob = swap_prob

    def _transform(self, text: str) -> list[str]:
        if not text:
            return [text]
        sentences = self.split(text)
        transformed_sentences = []
        i = 0
        while True:
            if i >= len(sentences):
                break
            if random.random() < self._swap_prob and i != len(sentences) - 1:
                sentences[i], sentences[i + 1] = sentences[i + 1], sentences[i]
                i += 2
            else:
                transformed_sentences.append(sentences[i])
                i += 1
        return [" ".join(transformed_sentences)]


class CopyStudentTransformation(StringTransformation):
    pass


class SentenceBlock(SentenceTransformation):

    MIN_NUM_CHARS_IN_SENTENCE = 32
    MAX_NUM_SENTENCES_IN_BLOCK = 12
    MIN_NUM_SENTENCES_IN_BLOCK = 1
    MAX_NUM_BLOCKS_PER_TEXT = 16

    def __init__(self, transformation_prob: float = 1) -> None:
        super().__init__(transformation_prob)

    def _merge_short_sentences(self, sentences: list[str]) -> list[str]:
        merged_sentences = []
        merged_sentence = ""
        for sentence in sentences:
            merged_sentence += " " + sentence
            if len(merged_sentence) >= self.MIN_NUM_CHARS_IN_SENTENCE:
                merged_sentences.append(merged_sentence.strip())
                merged_sentence = ""
        if merged_sentence:
            if len(merged_sentence) < self.MIN_NUM_CHARS_IN_SENTENCE:
                merged_sentences[-1] += " " + merged_sentence
            else:
                merged_sentences.append(merged_sentence)
        return merged_sentences

    def _group_into_blocks(self, sentences: list[str]) -> list[str]:
        sentences = self._merge_short_sentences(sentences)
        if len(sentences) < 2:
            sentences = [sentences[0][: len(sentences[0]) // 2], sentences[0][len(sentences[0]) // 2 :]]
        num_sentences = len(sentences)
        max_num_sentences_in_block = min(self.MAX_NUM_SENTENCES_IN_BLOCK, num_sentences - 1)
        block_sizes = []
        while True:
            block_sizes.append(random.randint(self.MIN_NUM_SENTENCES_IN_BLOCK, max_num_sentences_in_block))
            if sum(block_sizes) >= num_sentences:
                break
        block_sizes[-1] = num_sentences - sum(block_sizes[:-1])
        random.shuffle(block_sizes)
        blocked_sentences = []
        remaining_sentences = sentences
        for block_size in block_sizes:
            blocked_sentences.append(" ".join(remaining_sentences[:block_size]))
            remaining_sentences = remaining_sentences[block_size:]
        assert not remaining_sentences
        return blocked_sentences[: self.MAX_NUM_BLOCKS_PER_TEXT]

    def _transform(self, text: str) -> list[str]:
        sentences = self.split(text)
        blocked_sentences = self._group_into_blocks(sentences)
        return blocked_sentences

    @staticmethod
    def random_sum_to(n: int) -> list[int]:
        """Computes n random numbers in the interval [0, n] such that their sum is exactly n."""
        sel = sorted(random.randint(0, n) for _ in range(n - 1))
        return [upper - lower for lower, upper in zip([0] + sel, sel + [n])]
