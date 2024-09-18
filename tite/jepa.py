import inspect
from typing import Any, Callable, NamedTuple

from torch import Tensor
from torch.nn import Module


class MaskCollatorBatch(NamedTuple):
    input_ids: Tensor
    ctxt_attn_mask: Tensor
    pred_attn_mask: Tensor
    metadata: dict[str, Any]


Encoder = Callable[[Any], Any]
Predictor = Callable[[Any, Any], Any]
LossFn = Callable[[Any, Any], Any]


def parse_kwargs(kwargs: dict[str, Any], module: Module) -> dict[str, Any]:
    valid_keys = inspect.signature(module.forward).parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valid_keys}


class JEPA(Module):
    def __init__(
        self,
        student: Encoder,
        teacher: Encoder,
        predictor: Predictor,
        loss: LossFn,
        return_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self._student = student
        self._teacher = teacher
        self._predictor = predictor
        self._loss = loss
        self._return_embeddings = return_embeddings

    def forward(self, input: dict, target: dict | None, **aux):
        # TODO kwargs should contain the aux for the predictor if necessary
        embx = self._student(**input)
        if target is None:
            emby = embx
        else:
            teacher_kwargs = parse_kwargs(aux, self._teacher)
            emby = self._teacher(**target, **teacher_kwargs)
        predictor_kwargs = parse_kwargs(aux, self._predictor)
        pred = self._predictor(embx, **predictor_kwargs)
        if not self._return_embeddings:
            return self._loss(pred, emby)
        return self._loss(pred, emby), embx, emby
