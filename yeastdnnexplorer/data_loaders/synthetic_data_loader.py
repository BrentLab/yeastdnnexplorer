from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import torch

from yeastdnnexplorer.probability_models.generate_data import (
    generate_gene_population,
    generate_binding_effects, 
    generate_pvalues, 
    generate_perturbation_effects, 
    GenePopulation
)

class SyntheticDataLoader(LightningDataModule):
    def __init__(
            self, 
            batch_size: int = 32,
            num_genes: int = 1000, 
            signal: List[int] = [0.1, 0.15, 0.2, 0.25, 0.3],
            n_sample: List[int] = [1, 1, 2, 2, 4],
            val_size: float = 0.1, 
            test_size: float = 0.1, 
            random_state: int = 42
        ):
        super().__init__()
        self.batch_size = batch_size
        self.num_genes = num_genes
        self.num_tfs = sum(n_sample) # sum of all n_sample is the number of TFs
        self.signal = signal
        self.n_sample = n_sample
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.final_data_tensor: torch.Tensor = None
        self.binding_effect_matrix = None
        self.perturbation_effect_matrix: Optional[TensorDataset] =  None
        self.val_dataset: Optional[TensorDataset] =  None
        self.test_dataset: Optional[TensorDataset] =  None

    def prepare_data(self) -> None:
        # this will be a list of length 10 with a GenePopulation object in each element
        gene_populations_list = []
        for signal_proportion, n_draws in zip(self.signal, self.n_sample):
            for _ in range(n_draws):
                gene_populations_list.append(generate_gene_population(self.num_genes, signal_proportion))

        # Generate binding data for each gene population
        binding_effect_list = [generate_binding_effects(gene_population)
                            for gene_population in gene_populations_list]


        # Calculate p-values for binding data
        binding_pvalue_list = [generate_pvalues(binding_data) for binding_data in binding_effect_list]

        binding_data_combined = [torch.stack((gene_population.labels, binding_effect, binding_pval), dim=1)
                                for gene_population, binding_effect, binding_pval
                                in zip (gene_populations_list, binding_effect_list, binding_pvalue_list)]

        # Stack along a new dimension (dim=1) to create a tensor of shape [num_genes, num_TFs, 3]
        binding_data_tensor = torch.stack(binding_data_combined, dim=1)

        perturbation_effects_list = [generate_perturbation_effects(binding_data_tensor)
                             for _ in range(sum(self.n_sample))]

        perturbation_pvalue_list = [generate_pvalues(perturbation_effects)
                            for perturbation_effects in perturbation_effects_list]
        
        # Convert lists to tensors if they are not already
        perturbation_effects_tensor = torch.stack(perturbation_effects_list, dim=1)
        perturbation_pvalues_tensor = torch.stack(perturbation_pvalue_list, dim=1)

        # Ensure perturbation data is reshaped to match [n_genes, n_tfs]
        # This step might need adjustment based on the actual shapes of your tensors.
        perturbation_effects_tensor = perturbation_effects_tensor.unsqueeze(-1)  # Adds an extra dimension for concatenation
        perturbation_pvalues_tensor = perturbation_pvalues_tensor.unsqueeze(-1)  # Adds an extra dimension for concatenation

        # Concatenate along the last dimension to form a [n_genes, n_tfs, 5] tensor
        self.final_data_tensor = torch.cat((binding_data_tensor, perturbation_effects_tensor, perturbation_pvalues_tensor), dim=2)

    def setup(self, stage: Optional[str] = None) -> None:
        self.binding_effect_matrix = self.final_data_tensor[:, :, 1]
        self.perturbation_effect_matrix = self.final_data_tensor[:, :, 3]

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
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=15, shuffle=False, persistent_workers=True)
