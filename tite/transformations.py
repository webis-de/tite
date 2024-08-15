import random
from typing import Any

import nltk
import torch
from torch import LongTensor, Tensor
from torch.nn import Module


class Transformation(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwds: Any) -> tuple[dict, dict]:
        raise NotImplementedError


class MLMMaskTokens(Transformation):
    def __init__(self, vocab_size: int, mask_id: int, cls_id: int, sep_id: int, mask_prob: float = 0.3) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._mask_id = mask_id
        self._cls_id = cls_id
        self._sep_id = sep_id
        self._mask_prob = mask_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        mlm_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._mask_prob
        mlm_mask = mlm_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        probability_matrix = torch.rand(attention_mask.shape, device=input_ids.device)
        mask_mask = mlm_mask & (probability_matrix < 0.8)
        mask_random = mlm_mask & (probability_matrix >= 0.8) & (probability_matrix < 0.9)
        input_ids = torch.where(mask_mask, self._mask_id, input_ids)
        input_ids = torch.where(
            mask_random, torch.randint(self._vocab_size, input_ids.shape, device=input_ids.device), input_ids
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}, {"mlm_mask": mlm_mask}


class MaskTokens(Transformation):
    def __init__(self, mask_id: int, cls_id: int, sep_id: int, mask_prob: float = 0.3) -> None:
        super().__init__()
        self._mask_id = mask_id
        self._cls_id = cls_id
        self._sep_id = sep_id
        self._mask_prob = mask_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        mlm_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._mask_prob
        mlm_mask = mlm_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        input_ids = torch.where(mlm_mask, self._mask_id, input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, {}


class DeleteTokens(Transformation):
    def __init__(self, pad_id: int, cls_id: int, sep_id: int, delete_prob: float = 0.3):
        super().__init__()
        self._pad_id = pad_id
        self._cls_id = cls_id
        self._sep_id = sep_id
        self._delete_prob = delete_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        delete_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._delete_prob
        delete_mask = delete_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        num_delete = delete_mask.sum(-1)
        num_tokens = input_ids.shape[1] - num_delete
        new_input_ids = torch.nn.utils.rnn.pad_sequence(
            torch.split(input_ids[~delete_mask], num_tokens.tolist()), batch_first=True, padding_value=self._pad_id
        )
        new_attention_mask = torch.nn.utils.rnn.pad_sequence(
            torch.split(attention_mask[~delete_mask], num_tokens.tolist()), batch_first=True
        )
        return {"input_ids": new_input_ids, "attention_mask": new_attention_mask}, {}


class InsertRandomTokens(Transformation):
    def __init__(self, vocab_size: int, pad_id: int, insert_prob: float = 0.05):
        super().__init__()
        self._pad_id = pad_id
        self._insert_prob = insert_prob
        self._vocab_size = vocab_size

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        batch_size, orig_seq_len = input_ids.shape
        num_non_zero = attention_mask.sum(-1)
        batch_idx, token_idx = attention_mask.nonzero(as_tuple=True)

        insert_mask = torch.rand((batch_size, orig_seq_len), device=input_ids.device) < self._insert_prob
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

    def __call__(self, text: str) -> str:
        raise NotImplementedError


class CharacterTransformation(StringTransformation):

    def encode(self, text: str) -> Tensor:
        return torch.tensor(bytearray(text.encode("utf-8")))

    def decode(self, encoded: Tensor) -> str:
        return bytes(bytearray(encoded.tolist())).decode("utf-8")


class SentenceTransformation(StringTransformation):

    def split(self, text: str) -> list[str]:
        return nltk.sent_tokenize(text)


class CharacterDelete(CharacterTransformation):

    def __init__(self, delete_prob: float = 0.3):
        self._delete_prob = delete_prob

    # def __call__(self, text: str) -> str:
    #     encoded = self.encode(text)
    #     delete_mask = torch.rand(encoded.shape) < self._delete_prob
    #     transformed_encoded = encoded[~delete_mask & (encoded <= 128)]
    #     transformed = self.decode(transformed_encoded)
    #     return transformed

    def __call__(self, text: str) -> str:
        transformed_text = ""
        for char in text:
            if random.random() > self._delete_prob:
                transformed_text += char
        return transformed_text


class CharacterInsert(CharacterTransformation):

    def __init__(self, insert_prob: float = 0.05):
        self._insert_prob = insert_prob

    # def __call__(self, text: str) -> str:
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

    def __call__(self, text: str) -> str:
        transformed_text = ""
        for char in text:
            transformed_text += char
            if random.random() < self._insert_prob:
                transformed_text += random.choice(text)
        return transformed_text


class CharacterSwapNeighboring(CharacterTransformation):

    def __init__(self, swap_prob: float = 0.05):
        self._swap_prob = swap_prob

    # def __call__(self, text: str) -> str:
    #     encoded = self.encode(text)
    #     swap_mask = torch.rand(encoded.shape[0] - 1) < self._swap_prob
    #     encoded_transformed = encoded.clone()
    #     encoded_transformed[:-1][swap_mask] = encoded[1:][swap_mask]
    #     encoded_transformed[1:][swap_mask] = encoded[:-1][swap_mask]
    #     transformed = self.decode(encoded_transformed)
    #     return transformed

    def __call__(self, text: str) -> str:
        transformed_text = ""
        i = 0
        while True:
            if i >= len(text):
                break
            if random.random() < self._swap_prob and i != len(text) - 1:
                transformed_text += text[i + 1] + text[i]
                i += 2
            else:
                transformed_text += text[i]
                i += 1
        return transformed_text


class SentenceDelete(SentenceTransformation):

    def __init__(self, delete_prob: float = 0.3):
        self._delete_prob = delete_prob

    def __call__(self, text: str) -> str:
        if not text:
            return text
        sentences = self.split(text)
        transformed_sentences = []
        keep = [random.random() >= self._delete_prob for _ in range(len(sentences))]
        if not any(keep):
            keep[random.randint(0, len(keep) - 1)] = True
        for sentence, keep_sentence in zip(sentences, keep):
            if keep_sentence:
                transformed_sentences.append(sentence)
        return " ".join(transformed_sentences)


class SentenceSwapNeighboring(SentenceTransformation):

    def __init__(self, swap_prob: float = 0.05):
        self._swap_prob = swap_prob

    def __call__(self, text: str) -> str:
        if not text:
            return text
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
        return " ".join(transformed_sentences)
