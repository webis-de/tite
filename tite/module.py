from pathlib import Path
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerBase, TensorType

from .datasets import GLUEDataModule
from .glue_module import GlueModule
from .jepa import JEPA, LossFn, Predictor
from .transformations import Transformation


class _DetachFromGrad(Module):
    def __init__(self, module: Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._module = module

    def forward(self, *args, **kwargs) -> Tensor:
        output = self._module(*args, **kwargs)
        assert isinstance(output, Tensor)
        return output.detach()


class TiteModule(LightningModule):
    def __init__(
        self,
        student: Module,
        teacher: Module | None,
        tokenizer: PreTrainedTokenizerBase,
        transformations: list[Transformation],
        predictor: Predictor,
        loss: LossFn,
        text_key: str = "text",
        max_length: int | None = None,
        detach_teacher_from_grad: bool = False,
    ) -> None:
        super().__init__()
        if teacher is None:
            teacher = student
        # ties weights for BERT models -- only works for teacher MLM and student BERT
        if hasattr(student, "tie_decoder_weights") and teacher is not None:
            student.tie_decoder_weights(teacher)
        self._student = student
        self._teacher = teacher
        self._tokenizer = tokenizer
        self._transforms = transformations
        self._predictor = predictor
        self._loss = loss
        self._text_key = text_key
        self._max_length = max_length
        if detach_teacher_from_grad:
            self._teacher = _DetachFromGrad(self._teacher)
        self._jepa = JEPA(self._student, self._teacher, predictor, loss)
        # Stores the state before the current validation step (or None if currently not in a validation step).
        self.pre_val_student_state = None

        self._tokens_seen = 0.0

    def on_validation_start(self) -> None:
        assert self.pre_val_student_state is None
        self.pre_val_student_state = self._student.state_dict()
        # Train on GLUE
        glue = GLUEDataModule(batch_size=32)
        glue_module = GlueModule(self._student.train(), self._tokenizer, glue.hparams.name)
        trainer = Trainer(
            logger=False,
            precision=(self.trainer.precision if self.trainer is not None else "bf16-mixed"),
            max_epochs=10,
            # callbacks=[EarlyStopping(glue_module._evaluation_metrics[0].__class__.__name__, mode="max", patience=1)],
            enable_checkpointing=False,
        )
        trainer.fit(glue_module, glue)
        self._student.to(self.device)
        metrics = trainer.logged_metrics
        for name, value in metrics.items():
            self.log(f"{glue.hparams.name}/{name}", value, on_step=False, on_epoch=True)

    def on_validation_end(self) -> None:
        assert self.pre_val_student_state is not None
        # Restore Model to before it was evaluated on GLUE
        self._student.load_state_dict(self.pre_val_student_state)
        self.pre_val_student_state = None

    def validation_step(self, batch: dict[str, Any] | None) -> None:
        return

    def training_step(self, batch: dict[str, Any]):
        tokenized = self._tokenizer(
            text=batch[self._text_key],
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            return_tensors=TensorType.PYTORCH,
            truncation=True,
            max_length=self._max_length,
        ).to(self.device)
        for transformation in self._transforms:
            transformed = transformation(**tokenized)[0]
        jepa_loss = self._jepa(transformed, tokenized, None)
        attention_mask = tokenized["attention_mask"]
        num_tokens = attention_mask.sum() if attention_mask is not None else tokenized["input_ids"].numel()
        self._tokens_seen += num_tokens
        self.log("tokens_seen", self._tokens_seen, on_step=True, reduce_fx="sum")
        self.log_dict({"loss": jepa_loss}, prog_bar=True, on_step=True)
        return jepa_loss

    def save_pretrained(self, save_path: str | Path) -> None:
        self._student.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)

    def on_save_checkpoint(self, *args, **kwargs) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.save_pretrained(save_path)

    # def configure_optimizers(self) -> Optimizer:
    #     return AdamW(self._student.parameters())
