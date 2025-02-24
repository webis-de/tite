import random
from abc import abstractmethod

import nltk
import torch
from nltk import PunktSentenceTokenizer

from .transformation import Transformation


class StringTransformation(Transformation):

    @abstractmethod
    def transform(self, text: str) -> str:
        pass

    def __call__(self, texts: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], dict]:
        transformed_texts = []
        for batch_idx, text in texts:
            if not text:
                transformed_texts.append((batch_idx, text))
                continue
            for transformed_text in self.transform(text):
                transformed_texts.append((batch_idx, transformed_text))
        return transformed_texts, {}


class CharacterDelete(StringTransformation):

    def __init__(self, delete_prob: float = 0.3, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self.delete_prob = delete_prob

    def transform(self, text: str) -> list[str]:
        transformed_text = "".join(char for char in text if random.random() > self.delete_prob)
        return [transformed_text]


class CharacterInsert(StringTransformation):

    def __init__(self, insert_prob: float = 0.05, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self.insert_prob = insert_prob

    def transform(self, text: str) -> list[str]:
        transformed_text = "".join(
            char if random.random() > self.insert_prob else char + random.choice(text) for char in text
        )
        return [transformed_text]


class CharacterSwapNeighboring(StringTransformation):

    def __init__(self, swap_prob: float = 0.05, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self.swap_prob = swap_prob

    def transform(self, text: str) -> list[str]:
        transformed_text = list(text)
        num_swaps = int(torch.distributions.Binomial(len(text), self.swap_prob).sample().item())
        for _ in range(num_swaps):
            i = random.randint(0, len(text) - 2)
            transformed_text[i], transformed_text[i + 1] = transformed_text[i + 1], transformed_text[i]
        return ["".join(transformed_text)]


class SentenceTransformation(StringTransformation):

    def __init__(self, transformation_prob: float = 1.0) -> None:
        super().__init__(transformation_prob)
        self.tokenizer: PunktSentenceTokenizer = nltk.load("tokenizers/punkt/english.pickle")
        assert isinstance(self.tokenizer, PunktSentenceTokenizer)

    def split(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)


class SentenceDelete(SentenceTransformation):

    def __init__(self, delete_prob: float = 0.3, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self.delete_prob = delete_prob

    def transform(self, text: str) -> list[str]:
        if not text:
            return [text]
        sentences = self.split(text)
        keep = [random.random() >= self.delete_prob for _ in range(len(sentences))]
        if not any(keep):
            keep[random.randint(0, len(keep) - 1)] = True
        return [" ".join(sentence for sentence, keep_sentence in zip(sentences, keep) if keep_sentence)]


class SentenceSwapNeighboring(SentenceTransformation):

    def __init__(self, swap_prob: float = 0.05, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self.swap_prob = swap_prob

    def transform(self, text: str) -> list[str]:
        if not text:
            return [text]
        sentences = self.split(text)
        transformed_sentences = []
        i = 0
        while True:
            if i >= len(sentences):
                break
            if random.random() < self.swap_prob and i != len(sentences) - 1:
                transformed_sentences.append(sentences[i + 1])
                transformed_sentences.append(sentences[i])
                i += 2
            else:
                transformed_sentences.append(sentences[i])
                i += 1
        return [" ".join(transformed_sentences)]


class SentenceOffset(SentenceTransformation):

    def __init__(self, min_offset: int = 3, max_offset: int = 7, transformation_prob: float = 1.0):
        super().__init__(transformation_prob)
        self.min_offset = min_offset
        self.max_offset = max_offset

    def transform(self, text: str) -> list[str]:
        if not text:
            return [text]
        sentences = self.split(text)
        max_offset = min(self.max_offset, len(sentences) - 1)
        min_offset = min(self.min_offset, max_offset)
        offset = random.randint(min_offset, max_offset)
        transformed_sentences = sentences[offset:]
        if not transformed_sentences:
            raise ValueError("No sentences left after offset.")
        return [" ".join(transformed_sentences)]


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

    def transform(self, text: str) -> list[str]:
        sentences = self.split(text)
        blocked_sentences = self._group_into_blocks(sentences)
        return blocked_sentences

    @staticmethod
    def random_sum_to(n: int) -> list[int]:
        """Computes n random numbers in the interval [0, n] such that their sum is exactly n."""
        sel = sorted(random.randint(0, n) for _ in range(n - 1))
        return [upper - lower for lower, upper in zip([0] + sel, sel + [n])]
