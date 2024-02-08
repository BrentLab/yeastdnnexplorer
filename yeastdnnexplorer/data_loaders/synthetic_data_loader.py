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
        # TODO get the in silico data, will need to generate as many tables as we have genes (each table is associated with one TF)
            # binding_effect col is for the binding dataset table
            # perturbation_effect col is for the perturbation dataset table
            # so will generate 1000 tables and then write logic to create our matrices from those two tables
        
        # gene_population = generate_gene_population(1000, 0.3)
        # population1_tf1_data = generate_perturbation_binding_data(gene_population, 0.0, 1.0, 3.0, 1.0, 1e-3, 0.5)
        # population1_tf1_data['regulator'] = 'TF1'

        # TODO transform data into form that model will use (need to turn into the two matrices as described)

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
