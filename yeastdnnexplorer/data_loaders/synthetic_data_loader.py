from collections.abc import Callable

import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from yeastdnnexplorer.probability_models.generate_data import (
    default_perturbation_effect_adjustment_function,
    generate_binding_effects,
    generate_gene_population,
    generate_perturbation_effects,
    generate_pvalues,
)
from yeastdnnexplorer.probability_models.relation_classes import Relation


class SyntheticDataLoader(LightningDataModule):
    """A class for a synthetic data loader that generates synthetic bindiing &
    perturbation effect data for training, validation, and testing a model This class
    contains all of the logic for generating and parsing the synthetic data, as well as
    splitting it into train, validation, and test sets It is a subclass of
    pytorch_lightning.LightningDataModule, which is similar to a regular PyTorch
    DataLoader but with added functionality for data loading."""

    def __init__(
        self,
        batch_size: int = 32,
        num_genes: int = 1000,
        signal: list[float] = [0.1, 0.2, 0.2, 0.4, 0.5],
        signal_mean: float = 3.0,
        n_sample: list[int] = [1, 2, 2, 4, 4],
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
        max_mean_adjustment: float = 0.0,
        adjustment_function: Callable[
            [torch.Tensor, float, float, float], torch.Tensor
        ] = default_perturbation_effect_adjustment_function,
        tf_relationships: dict[int, list[int] | list[Relation]] = {},
    ) -> None:
        """
        Constructor of SyntheticDataLoader.

        :param batch_size: The number of samples in each mini-batch
        :type batch_size: int
        :param num_genes: The number of genes in the synthetic data (this is the number
            of datapoints in our dataset)
        :type num_genes: int
        :param signal: The proportion of genes in each sample group that are put in the
            signal grop (i.e. have a non-zero binding effect and expression response)
        :type signal: List[int]
        :param n_sample: The number of samples to draw from each signal group
        :type n_sample: List[int]
        :param val_size: The proportion of the dataset to include in the validation
            split
        :type val_size: float
        :param test_size: The proportion of the dataset to include in the test split
        :type test_size: float
        :param random_state: The random seed to use for splitting the data (keep this
            consistent to ensure reproduceability)
        :type random_state: int
        :param signal_mean: The mean of the signal distribution
        :type signal_mean: float
        :param max_mean_adjustment: The maximum mean adjustment to apply to the mean
                                    of the signal (bound) perturbation effects
        :type max_mean_adjustment: float
        :param adjustment_function: A function that adjusts the mean of the signal
                                    (bound) perturbation effects
        :type adjustment_function: Callable[[torch.Tensor, float, float,
                                   float, dict[int, list[int]]], torch.Tensor]
        :raises TypeError: If batch_size is not an positive integer
        :raises TypeError: If num_genes is not an positive integer
        :raises TypeError: If signal is not a list of integers or floats
        :raises TypeError: If n_sample is not a list of integers
        :raises TypeError: If val_size is not a float between 0 and 1 (inclusive)
        :raises TypeError: If test_size is not a float between 0 and 1 (inclusive)
        :raises TypeError: If random_state is not an integer
        :raises TypeError: If signal_mean is not a float
        :raises ValueError: If val_size + test_size is greater than 1 (i.e. the splits
            are too large)

        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise TypeError("batch_size must be a positive integer")
        if not isinstance(num_genes, int) or num_genes < 1:
            raise TypeError("num_genes must be a positive integer")
        if not isinstance(signal, list) or not all(
            isinstance(x, (int, float)) for x in signal
        ):
            raise TypeError("signal must be a list of integers or floats")
        if not isinstance(n_sample, list) or not all(
            isinstance(x, int) for x in n_sample
        ):
            raise TypeError("n_sample must be a list of integers")
        if not isinstance(val_size, (int, float)) or val_size <= 0 or val_size >= 1:
            raise TypeError("val_size must be a float between 0 and 1 (inclusive)")
        if not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1:
            raise TypeError("test_size must be a float between 0 and 1 (inclusive)")
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")
        if not isinstance(signal_mean, float):
            raise TypeError("signal_mean must be a float")
        if test_size + val_size > 1:
            raise ValueError("val_size + test_size must be less than or equal to 1")

        super().__init__()
        self.batch_size = batch_size
        self.num_genes = num_genes
        self.signal_mean = signal_mean
        self.signal = signal or [0.1, 0.15, 0.2, 0.25, 0.3]
        self.n_sample = n_sample or [1 for _ in range(len(self.signal))]
        self.num_tfs = sum(self.n_sample)  # sum of all n_sample is the number of TFs
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        self.max_mean_adjustment = max_mean_adjustment
        self.adjustment_function = adjustment_function
        self.tf_relationships = tf_relationships

        self.final_data_tensor: torch.Tensor = None
        self.binding_effect_matrix: torch.Tensor | None = None
        self.perturbation_effect_matrix: torch.Tensor | None = None
        self.val_dataset: TensorDataset | None = None
        self.test_dataset: TensorDataset | None = None

    def prepare_data(self) -> None:
        """Function to generate the raw synthetic data and save it in a tensor For
        explanations of the functions used to generate the data, see the
        generate_in_silico_data tutorial notebook in the docs No assertion checks are
        performed as that is handled in the functions in generate_data.py."""
        # this will be a list of length 10 with a GenePopulation object in each element
        gene_populations_list = []
        for signal_proportion, n_draws in zip(self.signal, self.n_sample):
            for _ in range(n_draws):
                gene_populations_list.append(
                    generate_gene_population(self.num_genes, signal_proportion)
                )

        # Generate binding data for each gene population
        binding_effect_list = [
            generate_binding_effects(gene_population)
            for gene_population in gene_populations_list
        ]

        # Calculate p-values for binding data
        binding_pvalue_list = [
            generate_pvalues(binding_data) for binding_data in binding_effect_list
        ]

        binding_data_combined = [
            torch.stack((gene_population.labels, binding_effect, binding_pval), dim=1)
            for gene_population, binding_effect, binding_pval in zip(
                gene_populations_list, binding_effect_list, binding_pvalue_list
            )
        ]

        # Stack along a new dimension (dim=1) to create a tensor of shape
        # [num_genes, num_TFs, 3]
        binding_data_tensor = torch.stack(binding_data_combined, dim=1)

        # if we are using a mean adjustment, we need to generate perturbation
        # effects in a slightly different way than if we are not using
        # a mean adjustment
        if self.max_mean_adjustment > 0:
            perturbation_effects_list = generate_perturbation_effects(
                binding_data_tensor,
                signal_mean=self.signal_mean,
                tf_index=0,  # unused
                max_mean_adjustment=self.max_mean_adjustment,
                adjustment_function=self.adjustment_function,
                tf_relationships=self.tf_relationships,
            )

            perturbation_pvalue_list = torch.zeros_like(perturbation_effects_list)
            for col_index in range(perturbation_effects_list.shape[1]):
                perturbation_pvalue_list[:, col_index] = generate_pvalues(
                    perturbation_effects_list[:, col_index]
                )

            # take absolute values
            perturbation_effects_list = torch.abs(perturbation_effects_list)

            perturbation_effects_tensor = perturbation_effects_list
            perturbation_pvalues_tensor = perturbation_pvalue_list
        else:
            perturbation_effects_list = [
                generate_perturbation_effects(
                    binding_data_tensor[:, tf_index, :].unsqueeze(1),
                    signal_mean=self.signal_mean,
                    tf_index=0,  # unused
                )
                for tf_index in range(sum(self.n_sample))
            ]
            perturbation_pvalue_list = [
                generate_pvalues(perturbation_effects)
                for perturbation_effects in perturbation_effects_list
            ]

            # take absolute values
            perturbation_effects_list = [
                torch.abs(perturbation_effects)
                for perturbation_effects in perturbation_effects_list
            ]

            # Convert lists to tensors
            perturbation_effects_tensor = torch.stack(perturbation_effects_list, dim=1)
            perturbation_pvalues_tensor = torch.stack(perturbation_pvalue_list, dim=1)

        # Ensure perturbation data is reshaped to match [n_genes, n_tfs]
        # This step might need adjustment based on the actual shapes of your tensors.
        perturbation_effects_tensor = perturbation_effects_tensor.unsqueeze(
            -1
        )  # Adds an extra dimension for concatenation
        perturbation_pvalues_tensor = perturbation_pvalues_tensor.unsqueeze(
            -1
        )  # Adds an extra dimension for concatenation

        # Concatenate along the last dimension to form a [n_genes, n_tfs, 5] tensor
        self.final_data_tensor = torch.cat(
            (
                binding_data_tensor,
                perturbation_effects_tensor,
                perturbation_pvalues_tensor,
            ),
            dim=2,
        )

    def setup(self, stage: str | None = None) -> None:
        """
        This function runs after prepare_data finishes and is used to split the data
        into train, validation, and test sets It ensures that these datasets are of the
        correct dimensionality and size to be used by the model.

        :param stage: The stage of the data setup (either 'fit' for training, 'validate'
            for validation, or 'test' for testing), unused for now as the model is not
            complicated enough to necessitate this
        :type stage: Optional[str]

        """
        self.binding_effect_matrix = self.final_data_tensor[:, :, 1]
        self.perturbation_effect_matrix = self.final_data_tensor[:, :, 3]

        # split into train, val, and test
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            self.binding_effect_matrix,
            self.perturbation_effect_matrix,
            test_size=(self.val_size + self.test_size),
            random_state=self.random_state,
        )

        # normalize test_size so that it is a percentage of the remaining data
        self.test_size = self.test_size / (self.val_size + self.test_size)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=self.test_size, random_state=self.random_state
        )

        # Convert to tensors
        X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
            Y_train, dtype=torch.float32
        )
        X_val, Y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(
            Y_val, dtype=torch.float32
        )
        X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(
            Y_test, dtype=torch.float32
        )

        # Set our datasets
        self.train_dataset = TensorDataset(X_train, Y_train)
        self.val_dataset = TensorDataset(X_val, Y_val)
        self.test_dataset = TensorDataset(X_test, Y_test)

    def train_dataloader(self) -> DataLoader:
        """
        Function to return the training dataloader, we shuffle to avoid learning based
        on the order of the data.

        :return: The training dataloader
        :rtype: DataLoader

        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=15,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Function to return the validation dataloader.

        :return: The validation dataloader
        :rtype: DataLoader

        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=15,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Function to return the testing dataloader.

        :return: The testing dataloader
        :rtype: DataLoader

        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=15,
            shuffle=False,
            persistent_workers=True,
        )
