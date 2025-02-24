import torch

from .teacher import Teacher


class BlockOrderTeacher(Teacher):

    def map_targets(self, input_ids: torch.Tensor, student_batch_idcs: tuple[int], **kwargs) -> torch.Tensor:
        block_size = torch.bincount(torch.tensor(student_batch_idcs, device=input_ids.device))
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.arange(bs, device=input_ids.device) for bs in block_size], batch_first=True, padding_value=-100
        )
        return targets
