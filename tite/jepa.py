from typing import Any, Mapping

from torch import Tensor
from torch.optim import AdamW, Optimizer
from pytorch_lightning import LightningModule


class TJEPA(LightningModule):
    """I-JEPA adapted for text"""

    def __init__(self, student: LightningModule, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.student = student

    def training_step(
        self, *args: Any, **kwargs: Any
    ) -> Tensor | Mapping[str, Any] | None:
        raise NotImplementedError()
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.student.parameters())
