import os

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class RealDataLoader(LightningDataModule):
    """
    A class to load in data from the CSV data for various binding and perturbation
    experiments.

    After loading in the data, the data loader will parse the data into the form
    expected by our models. It will also split the data into training, testing, and
    validation sets for the model to use.

    """

    def __init__(
        self,
        batch_size: int = 32,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
        data_dir_path: str | None = None,
    ) -> None:
        """
        Constructor of RealDataLoader.

        :param batch_size: The number of samples in each mini-batch
        :type batch_size: int
        :param val_size: The proportion of the dataset to include in the validation
            split
        :type val_size: float
        :param test_size: The proportion of the dataset to include in the test split
        :type test_size: float
        :param random_state: The random seed to use for splitting the data (keep this
            consistent to ensure reproduceability)
        :type random_state: int
        :param data_dir_path: The path to the directory containing the CSV files for the
            binding and perturbation data
        :type data_dir_path: str
        :raises TypeError: If batch_size is not an positive integer
        :raises TypeError: If val_size is not a float between 0 and 1 (inclusive)
        :raises TypeError: If test_size is not a float between 0 and 1 (inclusive)
        :raises TypeError: If random_state is not an integer
        :raises ValueError: If val_size + test_size is greater than 1 (i.e. the splits
            are too large)
        :raises ValueError: if no data_dir is provided
        :raises AssertinoError: if the dataset sizes do not match up after reading in
            the data from the CSV files

        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise TypeError("batch_size must be a positive integer")
        if not isinstance(val_size, (int, float)) or val_size <= 0 or val_size >= 1:
            raise TypeError("val_size must be a float between 0 and 1 (inclusive)")
        if not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1:
            raise TypeError("test_size must be a float between 0 and 1 (inclusive)")
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")
        if data_dir_path is None:
            raise ValueError("data_dir_path must be provided")
        if test_size + val_size > 1:
            raise ValueError("val_size + test_size must be less than or equal to 1")

        super().__init__()
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.data_dir_path = data_dir_path

        self.final_data_tensor: torch.Tensor = None
        self.binding_effect_matrix: torch.Tensor | None = None
        self.perturbation_effect_matrix: torch.Tensor | None = None
        self.val_dataset: TensorDataset | None = None
        self.test_dataset: TensorDataset | None = None

    def prepare_data(self) -> None:
        """
        This function reads in the binding data and perturbation data from the CSV files
        that we have for these datasets.

        It throws out any genes that are not present in both the binding and
        perturbation sets, and then structures the data in a way that the model expects
        and can use

        """

        brent_cc_path = os.path.join(self.data_dir_path, "binding/brent_nf_cc")
        brent_nf_csv_files = [
            f for f in os.listdir(brent_cc_path) if f.endswith(".csv")
        ]
        perturb_dataset_path = os.path.join(
            self.data_dir_path, "perturbation/hu_reimann_tfko"
        )
        perturb_dataset_csv_files = [
            f for f in os.listdir(perturb_dataset_path) if f.endswith(".csv")
        ]

        # get a list of the genes in the binding data csvs
        # for brent_cc (and the 3 perturb response datasets) the genes are
        # in the same order in each csv, so it suffices to grab the target_locus_tag
        # column from the first one
        brent_cc_genes_ids = pd.read_csv(
            os.path.join(brent_cc_path, brent_nf_csv_files[0])
        )["target_locus_tag"]
        perturb_dataset_genes_ids = pd.read_csv(
            os.path.join(perturb_dataset_path, perturb_dataset_csv_files[0])
        )["target_locus_tag"]

        # Get the intersection of the genes in the binding and perturbation data
        common_genes = set(brent_cc_genes_ids).intersection(perturb_dataset_genes_ids)

        # Read in binding data from csv files
        binding_data_effects = pd.DataFrame()
        binding_data_pvalues = pd.DataFrame()
        for i, file in enumerate(brent_nf_csv_files):
            file_path = os.path.join(brent_cc_path, file)
            df = pd.read_csv(file_path)

            # only keep the genes that are in the intersection
            # of the genes in the binding and perturbation data
            df = df[df["target_locus_tag"].isin(common_genes)]

            # we need to handle duplicates now
            # (some datasets have multiple occurrences of the same gene)
            # we will keep the occurrence with the highest value in the 'effect' column
            # we can do this by sorting the dataframe by the 'effect' column
            # in descending order and keeping the fist occurrence of each gene
            # this does require us to do some additional work later (see how we
            # are consistently setting the index to 'target_locus_tag',
            # this ensures all of our datasets are in the same order)
            df = df.sort_values("effect", ascending=False).drop_duplicates(
                subset="target_locus_tag", keep="first"
            )

            # on the first iteration, add target_locus_tag column to the binding data
            if i == 0:
                binding_data_effects["target_locus_tag"] = df["target_locus_tag"]
                binding_data_pvalues["target_locus_tag"] = df["target_locus_tag"]
                binding_data_effects.set_index("target_locus_tag", inplace=True)
                binding_data_pvalues.set_index("target_locus_tag", inplace=True)

            binding_data_effects[file] = df.set_index("target_locus_tag")["effect"]
            binding_data_pvalues[file] = df.set_index("target_locus_tag")["pvalue"]

        # Read in perturbation data from csv files
        perturbation_effects = pd.DataFrame()
        perturbation_pvalues = pd.DataFrame()
        for i, file in enumerate(perturb_dataset_csv_files):
            file_path = os.path.join(perturb_dataset_path, file)
            df = pd.read_csv(file_path)

            # only keep the genes that are in the
            # intersection of the genes in the binding and perturbation data
            df = df[df["target_locus_tag"].isin(common_genes)]

            # handle duplicates
            df = df.sort_values("effect", ascending=False).drop_duplicates(
                subset="target_locus_tag", keep="first"
            )

            # on the first iteration, add the target_locus_tag
            # column to the perturbation data
            if i == 0:
                perturbation_effects["target_locus_tag"] = df["target_locus_tag"]
                perturbation_pvalues["target_locus_tag"] = df["target_locus_tag"]
                perturbation_effects.set_index("target_locus_tag", inplace=True)
                perturbation_pvalues.set_index("target_locus_tag", inplace=True)

            perturbation_effects[file] = df.set_index("target_locus_tag")["effect"]
            perturbation_pvalues[file] = df.set_index("target_locus_tag")["pvalue"]

        # shapes should be equal at this point
        assert binding_data_effects.shape == perturbation_effects.shape
        assert binding_data_pvalues.shape == perturbation_pvalues.shape

        # reindex so that the rows in binding and perturb data match up
        # (we need genes to be in the same order)
        perturbation_effects = perturbation_effects.reindex(binding_data_effects.index)
        perturbation_pvalues = perturbation_pvalues.reindex(binding_data_pvalues.index)

        # concat the data into the shape expected by the model
        # we need to first convert the data to tensors
        binding_data_effects_tensor = torch.tensor(
            binding_data_effects.values, dtype=torch.float64
        )
        binding_data_pvalues_tensor = torch.tensor(
            binding_data_pvalues.values, dtype=torch.float64
        )
        perturbation_effects_tensor = torch.tensor(
            perturbation_effects.values, dtype=torch.float64
        )
        perturbation_pvalues_tensor = torch.tensor(
            perturbation_pvalues.values, dtype=torch.float64
        )

        # note that we no longer have a signal / noise tensor
        # (like for the synthetic data)
        self.final_data_tensor = torch.stack(
            [
                binding_data_effects_tensor,
                binding_data_pvalues_tensor,
                perturbation_effects_tensor,
                perturbation_pvalues_tensor,
            ],
            dim=-1,
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
        self.binding_effect_matrix = self.final_data_tensor[:, :, 0]
        self.perturbation_effect_matrix = self.final_data_tensor[:, :, 2]

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
