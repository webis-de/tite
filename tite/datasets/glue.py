from .basehfdatamodule import BaseHFDataModule

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
    def __init__(self, task: str, **kwargs) -> None:
        kwargs["text_keys"] = TASK_COLUMN_NAMES[task]
        super().__init__(path="glue", name=task, **kwargs)
