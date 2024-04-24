from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchmetrics import MeanAbsolutePercentageError, R2Score, MeanAbsoluteError
from yeastdnnexplorer.ml_models.metrics import SMSE

class SimpleModel(pl.LightningModule):
    """A class for a simple linear model that takes in binding effects for each
    transcription factor and predicts gene expression values This class contains all of
    the logic for setup, training, validation, and testing of the model, as well as
    defining how data is passed through the model It is a subclass of
    pytorch_lightning.LightningModule, which is similar to a regular PyTorch nn.module
    but with added functionality for training and validation."""

    def __init__(self, input_dim: int, output_dim: int, lr: float = 0.001) -> None:
        """
        Constructor of SimpleModel.

        :param input_dim: The number of input features to our model, these are the
            binding effects for each transcription factor for a specific gene
        :type input_dim: int
        :param output_dim: The number of output features of our model, this is the
            predicted gene expression value for each TF
        :type output_dim: int
        :param lr: The learning rate for the optimizer
        :type lr: float
        :raises TypeError: If input_dim is not an integer
        :raises TypeError: If output_dim is not an integer
        :raises TypeError: If lr is not a positive float
        :raises ValueError: If input_dim or output_dim are not positive

        """
        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be an integer")
        if not isinstance(output_dim, int):
            raise TypeError("output_dim must be an integer")
        if not isinstance(lr, float) or lr <= 0:
            raise TypeError("lr must be a positive float")
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive integers")

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.save_hyperparameters()

        self.mape = MeanAbsolutePercentageError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        self.SMSE = SMSE()

        # define layers for the model here
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model (i.e. how predictions are made for a given input)

        :param x: The input data to the model (minus the target y values)
        :type x: torch.Tensor
        :return: The predicted y values for the input x, this is a tensor of shape
            (batch_size, output_dim)
        :rtype: torch.Tensor

        """
        return self.linear1(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step for the model, this is called for each batch of data during
        training.

        :param batch: The batch of data to train on
        :type batch: Any
        :param batch_idx: The index of the batch
        :type batch_idx: int
        :return: The loss for the training batch
        :rtype: torch.Tensor

        """
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        self.log("train_mae", self.mae(y_pred, y))
        self.log("train_smse", self.SMSE(y_pred, y))
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Validation step for the model, this is called for each batch of data during
        validation.

        :param batch: The batch of data to validate on
        :type batch: Any
        :param batch_idx: The index of the batch
        :type batch_idx: int
        :return: The loss for the validation batch
        :rtype: torch.Tensor

        """
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)

        self.log("val_loss", loss)
        self.log("val_mae", self.mae(y_pred, y))
        self.log("val_smse", self.SMSE(y_pred, y))
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Test step for the model, this is called for each batch of data during testing
        Testing is only performed after training and validation when we have chosen a
        final model We want to test our final model on unseen data (which is why we use
        validation sets to "test" during training)

        :param batch: The batch of data to test on (this will have size (batch_size,
            input_dim)
        :type batch: Any
        :param batch_idx: The index of the batch
        :type batch_idx: int
        :return: The loss for the test batch
        :rtype: torch.Tensor

        """
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log("test_loss", loss)
        self.log("test_mae", self.mae(y_pred, y))
        self.log("test_smse", self.SMSE(y_pred, y))
        return loss

    def configure_optimizers(self) -> Optimizer:
        """
        Configure the optimizer for the model.

        :return: The optimizer for the model
        :rtype: Optimizer

        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
