from transformers import PreTrainedTokenizerBase

from .basehfdatamodule import BaseHFDataModule, Collator

TASK_COLUMN_NAMES = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}


class GLUEDataModule(BaseHFDataModule):
    def __init__(
        self, task: str, tokenizer: PreTrainedTokenizerBase, add_special_tokens: bool = True, **kwargs
    ) -> None:
        text_keys = TASK_COLUMN_NAMES[task]
        collator = Collator(tokenizer, text_keys, max_length=256)
        super().__init__(path="glue", name=task, collator=collator, **kwargs)
