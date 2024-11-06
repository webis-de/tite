import os
from pathlib import Path
from typing import Any, Literal

import torch
from lightning import LightningModule, Trainer
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override

from tite.datasets import FineWebDataModule  # noqa
from tite.lr_schedulers import LR_SCHEDULERS, WarmupLRScheduler
from tite.model import PreTrainedModel, TiteConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomSaveConfigCallback(SaveConfigCallback):
    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit" or trainer.logger is None:
            return
        return super().setup(trainer, pl_module, stage)


class CustomWandbLogger(WandbLogger):
    def __init__(
        self,
        name: str | None = None,
        save_dir: str | Path = ".",
        version: str | None = None,
        offline: bool = False,
        dir: str | Path | None = None,
        id: str | None = None,
        anonymous: bool | None = None,
        project: str | None = None,
        log_model: bool | Literal["all"] = False,
        experiment: Any | None = None,
        prefix: str = "",
        checkpoint_name: str | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            name,
            save_dir,
            version,
            offline,
            dir,
            id,
            anonymous,
            project,
            log_model,
            experiment,
            prefix,
            checkpoint_name,
            settings={"sync_dir": None, "sync_file": None, "ignore_globs": ["*.ckpt", "*.safetensors"]},
            **kwargs
        )

    @property
    def save_dir(self) -> str | None:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """
        if isinstance(self.experiment, DummyExperiment):
            return None
        return self.experiment.dir


class CustomLightningCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupLRScheduler | None = None,
    ) -> Any:
        if lr_scheduler is None:
            return optimizer

        return [optimizer], [{"scheduler": lr_scheduler, "interval": lr_scheduler.interval}]

    def add_arguments_to_parser(self, parser):

        def compute_global_batch_size(accumulate_grad_batches: int, batch_size: int) -> int:
            return accumulate_grad_batches * batch_size

        parser.add_lr_scheduler_args(tuple(LR_SCHEDULERS))
        parser.link_arguments("trainer.max_steps", "lr_scheduler.init_args.num_training_steps")
        parser.link_arguments(
            ("trainer.accumulate_grad_batches", "data.init_args.batch_size"),
            "lr_scheduler.init_args.batch_size",
            compute_fn=compute_global_batch_size,
        )


class DummyTite(PreTrainedModel):
    def __init__(self, config: TiteConfig):
        super().__init__(config)
        self.config = config
        self.param = torch.nn.Parameter(torch.Tensor([42]), requires_grad=True)
        self.register_parameter("super important parameter", self.param)
        self.post_init()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], self.config.last_hidden_size)) + self.param


def main():
    """
    generate config using `python main.py fit --print_config > config.yaml`
    additional callbacks at:
    https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks

    Example:
        To obtain a default config:

            python main.py fit \
                --trainer.callbacks=ModelCheckpoint \
                --optimizer AdamW \
                --trainer.logger CustomWandbLogger \
                --print_config > default.yaml

        To run with the default config:

            python main.py fit \
                --config default.yaml

    """
    CustomLightningCLI(
        save_config_callback=CustomSaveConfigCallback,
        save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True},
    )


if __name__ == "__main__":
    main()
