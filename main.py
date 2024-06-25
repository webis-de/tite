from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override
from tite.model import TiteModule  # noqa
from tite.datasets import FineWebDataModule  # noqa

# from lightning_ir.lightning_utils.warmup_schedulers import (
#     LR_SCHEDULERS,
#     WarmupScheduler,
# )

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


class CustomSaveConfigCallback(SaveConfigCallback):
    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit" or trainer.logger is None:
            return
        return super().setup(trainer, pl_module, stage)


class CustomWandbLogger(WandbLogger):
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
    # @staticmethod
    # def configure_optimizers(
    #     lightning_module: LightningModule,
    #     optimizer: torch.optim.Optimizer,
    #     lr_scheduler: WarmupScheduler | None = None,
    # ) -> Any:
    #     if lr_scheduler is None:
    #         return optimizer

    #     return [optimizer], [
    #         {"scheduler": lr_scheduler, "interval": lr_scheduler.interval}
    #     ]

    def add_arguments_to_parser(self, parser):
        pass
        # parser.add_lr_scheduler_args(tuple(LR_SCHEDULERS))
        # parser.link_arguments(
        #     "model.init_args.model_name_or_path", "data.init_args.model_name_or_path"
        # )
        # parser.link_arguments("model.init_args.config", "data.init_args.config")
        # parser.link_arguments("model", "data.init_args.model", apply_on="instantiate")
        # parser.link_arguments(
        #     "trainer.max_steps", "lr_scheduler.init_args.num_training_steps"
        # )


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
