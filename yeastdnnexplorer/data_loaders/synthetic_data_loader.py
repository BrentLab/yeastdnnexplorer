from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import torch

from yeastdnnexplorer.probability_models.generate_data import (generate_gene_population, generate_perturbation_binding_data)

class SyntheticDataLoader(LightningDataModule):
    def __init__(
            self, 
            batch_size: int = 32,
            num_genes: int = 1000, 
            num_tfs: int = 4, 
            val_size: float = 0.1, 
            test_size: float = 0.1, 
            random_state: int = 42
        ):
        super().__init__()
        self.batch_size = batch_size
        self.num_genes = num_genes
        self.num_tfs = num_tfs
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.all_raw_data = []
        self.binding_effect_matrix = None
        self.perturbation_effect_matrix: Optional[TensorDataset] =  None
        self.val_dataset: Optional[TensorDataset] =  None
        self.test_dataset: Optional[TensorDataset] =  None

    def prepare_data(self) -> None:
        for i in range(self.num_tfs):
            # load in the in silico data for this tf
            gene_population = generate_gene_population(self.num_genes, 0.3)
            population_data = generate_perturbation_binding_data(gene_population, 0.0, 1.0, 3.0, 1.0, 1e-3, 0.5)
            population_data['regulator'] = f'TF{i}'

            # we are dropping the gene_id column for now because tensors do not support non-numeric data
            # we can also drop the regulator column because we know which TF it corresponds to (the z-index in final tensor will be the TF index)

            self.all_raw_data.append(population_data.drop(['regulator', 'gene_id'], axis=1).to_numpy(dtype=np.float32))


    def setup(self, stage: Optional[str] = None) -> None:
        # we set up our data in this method (convert self.all_raw_data into the two matrices that the model will use)
        stacked_array = np.stack(self.all_raw_data, axis=0)
        tensor_3d = torch.tensor(stacked_array, dtype=torch.float32)

        self.binding_effect_matrix = [[0 for _ in range(self.num_tfs)] for _ in range(self.num_genes)] # rows will be genes, cols will be TFs, values will be binding effect
        self.perturbation_effect_matrix = [[0 for _ in range(self.num_tfs)] for _ in range(self.num_genes)] # rows will be genes, cols will be TFs, values will be perturbation effect

        # TODO this shouldn't be hardcoded (needs to be checked in some way)
        binding_effect_col_index = 3
        perturbation_effect_col_index = 1

        for tf in range(self.num_tfs):
            for gene in range(self.num_genes): 
                self.binding_effect_matrix[gene][tf] = tensor_3d[tf][gene][binding_effect_col_index].item()

                # use absolute value because we are only interested in if it changed, not whether it went up or down
                self.perturbation_effect_matrix[gene][tf] = abs(tensor_3d[tf][gene][perturbation_effect_col_index].item())

        # split into train, val, and test
        X_train, X_temp, Y_train, Y_temp = train_test_split(self.binding_effect_matrix, self.perturbation_effect_matrix, test_size=(self.val_size + self.test_size), random_state=self.random_state)

        # normalize test_size so that it is a percentage of the remaining data
        self.test_size = self.test_size / (self.val_size + self.test_size)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=self.test_size, random_state=self.random_state)

        # Convert to tensors
        X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)
        X_val, Y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)
        X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)

        # Set our datasets
        self.train_dataset = TensorDataset(X_train, Y_train)
        self.val_dataset = TensorDataset(X_val, Y_val)
        self.test_dataset = TensorDataset(X_test, Y_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=15, shuffle=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15, shuffle=False, persistent_workers=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15, shuffle=False, persistent_workers=True)
