from lightning.pytorch.profilers import PyTorchProfiler
from lightning_ir import (
    BiEncoderConfig,
    BiEncoderModule,
    FLOPSRegularization,
    KLDivergence,
    LightningIRDataModule,
    LightningIRTrainer,
    RunDataset,
    ScoreBasedInBatchCrossEntropy,
    SpladeConfig,
    SupervisedMarginMSE,
)
from torch.optim import AdamW

from tite.utils.callbacks import DummyImportCallback

profiler = PyTorchProfiler(
    sort_by_key="cpu_time",
    export_to_chrome=False,
    record_module_names=False,
    # group_by_input_shapes=True,
    # record_shapes=True,
)
trainer = LightningIRTrainer(
    precision="bf16-mixed",
    logger=False,
    profiler=profiler,
    max_steps=100,
    num_sanity_val_steps=0,
    enable_checkpointing=False,
)

data_module = LightningIRDataModule(
    num_workers=4,
    train_batch_size=8,
    train_dataset=RunDataset(
        "/mnt/ceph/storage/data-tmp/current/fschlatt/lightning-ir-experiments/runs-archive/"
        "__10000__msmarco-passage-train-judged.run",
        depth=100,
        sample_size=8,
        sampling_strategy="log_random",
        targets="score",
    ),
)

model = BiEncoderModule(
    model_name_or_path="./models/pre-trained/tite-2-late-flash-new",
    config=SpladeConfig(
        similarity_function="dot",
        doc_length=256,
        tie_projection=False,
        query_pooling_strategy="max",
        doc_pooling_strategy="max",
    ),
    # config=BiEncoderConfig(doc_length=256),
    loss_functions=[
        (SupervisedMarginMSE(), 0.05),
        KLDivergence(),
        ScoreBasedInBatchCrossEntropy(min_target_diff=3),
        FLOPSRegularization(query_weight=0.01, doc_weight=0.02),
    ],
)
model.set_optimizer(AdamW, lr=1e-5)

trainer.fit(model, data_module)
