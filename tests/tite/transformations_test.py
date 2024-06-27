import unittest
from unittest.mock import MagicMock, create_autospec, patch

import torch
from transformers import TensorType

from tite.module import TiteModule
from tite.transformations import SwapTokens


class SwapTokensTest(unittest.TestCase):

    def test_integration(self):
        student = MagicMock(return_value=torch.rand(3, 768))
        tokenizer_out = {
            "input_ids": torch.rand(3, 5),
            "attention_mask": torch.Tensor(
                [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
            ),
        }
        tokenizer = MagicMock(return_value=tokenizer_out)
        transform = SwapTokens()
        batch = {"text": ["hello", "world", ""], "misc": [42, 42, 42]}
        module = TiteModule(student, tokenizer, transform, "text")
        loss = module.training_step(batch)
        self.assertIsInstance(loss, torch.Tensor)
