import unittest
from unittest.mock import MagicMock

import torch
from transformers import TensorType

from tite.module import TiteModule


class TiteModuleTest(unittest.TestCase):
    def test_training_step(self):
        student = MagicMock(return_value=torch.rand(3, 768))
        tokenizer_out = {"input_ids": torch.rand(3, 5)}
        transform_out = [{"input_ids": torch.rand(3, 5)}]
        tokenizer = MagicMock(return_value=tokenizer_out)
        transform = MagicMock(return_value=transform_out)
        batch = {"text": ["hello", "world", ""], "misc": [42, 42, 42]}
        module = TiteModule(student, tokenizer, transform, "text")
        loss = module.training_step(batch)
        tokenizer.assert_called_once_with(
            text=["hello", "world", ""],
            return_attention_mask=True,
            padding=True,
            return_tensors=TensorType.PYTORCH,
        )
        transform.assert_called_once_with(**tokenizer_out)
        for tout in transform_out:
            student.assert_called_with(**tout)
        self.assertIsInstance(loss, torch.Tensor)
