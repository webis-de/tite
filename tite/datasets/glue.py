from .basehfdatamodule import BaseHFDataModule

TASK_COLUMN_NAMES = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}


class GLUEDataModule(BaseHFDataModule):
    def __init__(self, task_name: str = "mrpc", **kwargs) -> None:
        kwargs["text_keys"] = TASK_COLUMN_NAMES[task_name]
        super().__init__(path="glue", name=task_name, **kwargs)
