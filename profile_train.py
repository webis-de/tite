import types

import torch
from lightning import Trainer
from lightning.pytorch.profilers import PyTorchProfiler

from tite.datasets import FineWebDataModule, TransformationCollator
from tite.model.tite import TiteConfig, TiteForPreTraining
from tite.model.tokenizer import TiteTokenizer
from tite.module import TiteModule

torch.set_float32_matmul_precision("medium")


profiler = PyTorchProfiler(
    sort_by_key="cpu_time_total",
    export_to_chrome=False,
    record_module_names=False,
    group_by_input_shapes=True,
    record_shapes=True,
)


config = TiteConfig(
    vocab_size=30522,
    num_hidden_layers=12,
    kernel_sizes=(None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2),
    strides=(None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2),
    hidden_sizes=768,
    num_attention_heads=12,
    intermediate_sizes=3072,
    positional_embedding_type="rotary",
    rotary_interleaved=True,
    hidden_act="gelu_pytorch_tanh",
    norm_location="pre",
    norm_type="rms",
)

tokenizer = TiteTokenizer(
    vocab_file="tokenizers/tite/vocab.txt",
    tokenizer_file="tokenizers/tite/tokenizer.json",
    do_lower_case=True,
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    mask_token="[MASK]",
    model_max_length=512,
)
model = TiteForPreTraining(config)

module = TiteModule(model, tokenizer)


def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        fused=True,
    )
    return optimizer


module.configure_optimizers = types.MethodType(configure_optimizers, module)


datamodule = FineWebDataModule(
    path="arrow",
    data_files={"train": "./HuggingFaceFW___fineweb-edu/default/0.0.0/*/fineweb-edu-train-*.arrow"},
    batch_size=8,
    collator=TransformationCollator(tokenizer=tokenizer, text_keys=("text", None)),
    num_workers=8,
)

trainer = Trainer(
    accelerator="gpu",
    precision="bf16-mixed",
    profiler=profiler,
    logger=False,
    enable_checkpointing=False,
    max_steps=200,
    num_sanity_val_steps=0,
)

trainer.fit(module, datamodule=datamodule)
