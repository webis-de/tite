from lightning import LightningDataModule


class FineWebDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        # TODO load fineweb data

    def prepare_data(self):
        # download, split, etc...
        pass

    def setup(self, stage=None):
        # build dataset
        pass

    def train_dataloader(self):
        # return DataLoader
        pass

    def val_dataloader(self):
        # return DataLoader
        pass
        # TODO add glue datasets

    def test_dataloader(self):
        # return DataLoader
        pass
