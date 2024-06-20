from transformers import PretrainedConfig

import torch


class TiteConfig(PretrainedConfig):
    pass


class TiteModel(torch.nn.Module):

    def __init__(self, config: TiteConfig):
        super().__init__()
        self.config = config
