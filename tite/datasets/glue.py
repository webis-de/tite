from .basehfdatamodule import BaseHFDataModule


class GLUEDataModule(BaseHFDataModule):

    def __init__(self, task_name: str = "mrpc", **kwargs) -> None:
        super().__init__(path="glue", name=task_name, **kwargs)
