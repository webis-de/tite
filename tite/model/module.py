from lightning import LightningModule

from .model import TiteModel


class TiteModule(LightningModule):

    def __init__(self, model: TiteModel):
        super().__init__()
        self.model = model

    def foreard(self, *args, **kwargs):
        return self.model(*args, **kwargs)
