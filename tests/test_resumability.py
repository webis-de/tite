"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from tite.datasets import FineWebDataModule

batches = []


class DummyModel(LightningModule):
    def training_step(self, batch):
        global batches
        print(batch)
        batches.append(batch)
        return torch.rand(1, requires_grad=True)

    def configure_optimizers(self):
        return super().configure_optimizers()


class TiteModuleTest(unittest.TestCase):
    def test_basic_resumability(self):
        datamodule = FineWebDataModule(batch_size=2, seed=42)
        datamodule.setup("fit")
        loader = datamodule.train_dataloader()
        iter = loader.__iter__()
        discarded = [next(iter) for _ in range(10)]
        # for _ in range(10):
        #    next(iter)
        state = datamodule.state_dict()
        values = [next(iter) for _ in range(10)]
        datamodule.teardown("fit")
        del datamodule

        # Reproduce from the state
        datamodule = FineWebDataModule(batch_size=2, seed=42)
        datamodule.setup("fit")
        loader = datamodule.train_dataloader()
        datamodule.load_state_dict(state)
        self.assertSequenceEqual([next(iter) for _ in range(10)], values)
        datamodule.teardown("fit")

    def test_resume_hf_dataset(self):
        with TemporaryDirectory() as tmpdir:
            global batches
            # Perform quick dummy run and store it in a checkpoint
            datamodule = FineWebDataModule(batch_size=2, seed=42)
            callbacks = [ModelCheckpoint(dirpath=Path(tmpdir), every_n_train_steps=1, save_top_k=-1)]
            trainer = Trainer(max_steps=5, enable_checkpointing=True, default_root_dir=tmpdir, callbacks=callbacks)
            with trainer.init_module():
                model = DummyModel()
            trainer.fit(model, datamodule=datamodule)
            del datamodule
            del trainer
            del model

            # Resume from the checkpoint
            prev_batches = batches
            batches = []
            datamodule = FineWebDataModule(batch_size=2, seed=42)
            trainer = Trainer(
                max_steps=5,
                default_root_dir=tmpdir,
            )
            with trainer.init_module():
                model = DummyModel()
            trainer.fit(model, datamodule=datamodule, ckpt_path=Path(tmpdir) / "epoch=0-step=3.ckpt")
            print(f"Num Elements: {len(prev_batches)}")
            self.assertSequenceEqual(prev_batches[3:], batches)

"""
