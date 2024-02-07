from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
import torch

from yeastdnnexplorer.probability_models.generate_data import (generate_gene_population, generate_perturbation_binding_data)

class MyDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        # Add any other initialization parameters you might need

    def prepare_data(self):
        pass        

    def setup(self, stage=None):
        # get the in silico data

        # transform data into form that model will use

        # Set our datasets
        # self.train_dataset = TensorDataset(...)
        # self.val_dataset = TensorDataset(...)
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    # TODO: do we need a predict_dataloader?
