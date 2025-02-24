from abc import abstractmethod

import torch
from transformers import BatchEncoding

from .transformation import Transformation


class TokenTransformation(Transformation):

    @abstractmethod
    def transform(self, encoding: BatchEncoding) -> tuple[BatchEncoding, dict]: ...

    def __call__(self, encoding: BatchEncoding) -> tuple[BatchEncoding, dict]:
        return self.transform(encoding)


class TokenMLMMask(TokenTransformation):
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
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.mask_prob = mask_prob

    def transform(self, encoding: BatchEncoding) -> tuple[BatchEncoding, dict]:
        input_ids = encoding.input_ids.clone()
        attention_mask = encoding.attention_mask
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        mlm_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self.mask_prob
        special_tokens_mask = input_ids.eq(self.cls_id).logical_or(input_ids.eq(self.sep_id))
        mlm_mask = mlm_mask.logical_and(~special_tokens_mask)
        mlm_mask = mlm_mask.logical_and(transformation_mask[:, None])
        probability_matrix = torch.rand(attention_mask.shape, device=input_ids.device)
        mask_mask = mlm_mask & (probability_matrix < 0.8)
        mask_random = mlm_mask & (probability_matrix >= 0.8) & (probability_matrix < 0.9)
        input_ids = torch.where(mask_mask, self.mask_id, input_ids)
        input_ids = torch.where(
            mask_random, torch.randint(self.vocab_size, input_ids.shape, device=input_ids.device), input_ids
        )
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask}), {
            "mlm_mask": mlm_mask,
            "special_tokens_mask": special_tokens_mask,
            "original_input_ids": encoding.input_ids,
        }


class TokenMask(TokenTransformation):
    def __init__(
        self, mask_id: int, cls_id: int, sep_id: int, mask_prob: float = 0.3, transformation_prob: float = 1.0
    ) -> None:
        super().__init__(transformation_prob)
        self.mask_id = mask_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.mask_prob = mask_prob

    def transform(self, encoding: BatchEncoding) -> tuple[BatchEncoding, dict]:
        input_ids = encoding.input_ids.clone()
        attention_mask = encoding.attention_mask
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        mlm_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self.mask_prob
        special_tokens_mask = input_ids.eq(self.cls_id).logical_or(input_ids.eq(self.sep_id))
        mlm_mask = mlm_mask.logical_and(~special_tokens_mask)
        mlm_mask = mlm_mask.logical_and(transformation_mask[:, None])
        input_ids = torch.where(mlm_mask, self.mask_id, input_ids)
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask}), {
            "mlm_mask": mlm_mask,
            "special_tokens_mask": special_tokens_mask,
            "original_input_ids": encoding.input_ids,
        }


class TokenReplace(TokenTransformation):

    def __init__(
        self,
        vocab_size: int,
        cls_id: int,
        sep_id: int,
        replace_prob: float = 0.05,
        transformation_prob: float = 1.0,
    ):
        super().__init__(transformation_prob)
        self.vocab_size = vocab_size
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.replace_prob = replace_prob

    def transform(self, encoding: BatchEncoding) -> tuple[BatchEncoding, dict]:
        input_ids = encoding.input_ids.clone()
        attention_mask = encoding.attention_mask
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        random_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self.replace_prob
        random_mask = random_mask.logical_and(input_ids != self.cls_id).logical_and(input_ids != self.sep_id)
        random_mask = random_mask.logical_and(transformation_mask[:, None])
        input_ids = torch.where(random_mask, torch.randint_like(input_ids, 0, self.vocab_size), input_ids)
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask}), {}


class TokenDelete(TokenTransformation):
    def __init__(
        self, pad_id: int, cls_id: int, sep_id: int, delete_prob: float = 0.3, transformation_prob: float = 1.0
    ):
        super().__init__(transformation_prob)
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.delete_prob = delete_prob

    def transform(self, encoding: BatchEncoding) -> tuple[BatchEncoding, dict]:
        input_ids = encoding.input_ids.clone()
        attention_mask = encoding.attention_mask
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        delete_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self.delete_prob
        delete_mask = delete_mask.logical_and(input_ids != self.cls_id).logical_and(input_ids != self.sep_id)
        delete_mask = delete_mask.logical_and(transformation_mask[:, None])
        num_delete = delete_mask.sum(-1)
        num_tokens = input_ids.shape[1] - num_delete
        new_input_ids = torch.nn.utils.rnn.pad_sequence(
            torch.split(input_ids[~delete_mask], num_tokens.tolist()), batch_first=True, padding_value=self.pad_id
        )
        new_attention_mask = torch.nn.utils.rnn.pad_sequence(
            torch.split(attention_mask[~delete_mask], num_tokens.tolist()), batch_first=True
        )
        return BatchEncoding({"input_ids": new_input_ids, "attention_mask": new_attention_mask}), {}


class TokenInsert(TokenTransformation):
    def __init__(self, vocab_size: int, pad_id: int, insert_prob: float = 0.05, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self.pad_id = pad_id
        self.insert_prob = insert_prob
        self.vocab_size = vocab_size

    def transform(self, encoding: BatchEncoding) -> tuple[BatchEncoding, dict]:
        input_ids = encoding.input_ids.clone()
        attention_mask = encoding.attention_mask
        transformation_mask = self.transformation_mask(input_ids.shape[0])
        batch_size, orig_seq_len = input_ids.shape
        num_non_zero = attention_mask.sum(-1)
        batch_idx, token_idx = attention_mask.nonzero(as_tuple=True)

        insert_mask = torch.rand((batch_size, orig_seq_len), device=input_ids.device) < self.insert_prob
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
            torch.randint(self.vocab_size, (batch_size, new_seq_len), device=input_ids.device),
            new_input_ids,
        )

        new_lengths = num_non_zero + num_added
        new_attention_mask = (
            torch.arange(new_seq_len, device=input_ids.device)[None].expand(batch_size, -1) < new_lengths[..., None]
        )
        new_input_ids[~new_attention_mask] = self.pad_id
        return BatchEncoding({"input_ids": new_input_ids, "attention_mask": new_attention_mask}), {}
