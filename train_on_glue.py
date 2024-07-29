from lightning import Trainer

from tite.datasets import GLUEDataModule
from tite.glue_module import GlueModule
from tite.model import TiteModel
from tite.tokenizer import TiteTokenizer

tite = TiteModel.from_pretrained("./wandb/latest-run/files/huggingface_checkpoint")
glue = GLUEDataModule(batch_size=32)
model = GlueModule(
    tite,
    TiteTokenizer(
        vocab_file="tokenizers/bert-base-uncased/vocab.txt",
        tokenizer_file="tokenizers/bert-base-uncased/tokenizer.json",
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        model_max_length=512,
    ),
    glue.hparams.name,
)
trainer = Trainer(precision="bf16-mixed", max_epochs=5, enable_checkpointing=False)
trainer.fit(model, glue)
