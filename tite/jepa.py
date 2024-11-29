import inspect
from typing import Any, Callable, NamedTuple

from torch import Tensor
from torch.nn import Module

from tite.model import TiteModelOutput
from tite.teacher import CopyStudent


class MaskCollatorBatch(NamedTuple):
    input_ids: Tensor
    ctxt_attn_mask: Tensor
    pred_attn_mask: Tensor
    metadata: dict[str, Any]


LossFn = Callable[[Any, Any], Any]


def parse_kwargs(kwargs: dict[str, Any], module: Module) -> dict[str, Any]:
    valid_keys = inspect.signature(module.forward).parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valid_keys}


class JEPA(Module):
    def __init__(
        self,
        student: Module,
        teachers: list[Module],
        predictors: list[Module],
        losses: list[LossFn],
    ) -> None:
        super().__init__()
        self.student = student
        self.teachers = teachers
        self.predictors = predictors
        self.losses = losses

    def forward(self, input: dict, target: dict | None, **aux):
        student_aux = {k[8:]: v for k, v in aux.items() if k.startswith("student_")}
        teacher_aux = {k[8:]: v for k, v in aux.items() if k.startswith("teacher_")}
        output = self.student(**input, output_hidden_states=True)
        if isinstance(output, TiteModelOutput):
            embx = output.last_hidden_state
        else:
            embx = output
        losses = {}
        for teacher, predictor, loss in zip(self.teachers, self.predictors, self.losses):
            if target is None or isinstance(teacher, CopyStudent):
                emby = embx.detach()
            else:
                teacher_kwargs = parse_kwargs({**student_aux, **teacher_aux}, teacher)
                emby = teacher(**target, **teacher_kwargs)
                if isinstance(emby, TiteModelOutput):
                    emby = emby.last_hidden_state
            predictor_kwargs = parse_kwargs({**student_aux, **teacher_aux, **(target or {})}, predictor)
            pred = predictor(embx, **predictor_kwargs)
            losses[loss.__class__.__name__] = loss(pred, emby)
        return losses, output
