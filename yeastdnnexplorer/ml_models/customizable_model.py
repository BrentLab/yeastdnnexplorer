from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer


class CustomizableModel(pl.LightningModule):
    """
    A class for a customizable model that takes in binding effects for each
    transcription factor and predicts gene expression values This class contains all of
    the logic for setup, training, validation, and testing of the model, as well as
    defining how data is passed through the model It is a subclass of
    pytorch_lightning.LightningModule, which is similar to a regular PyTorch nn.module
    but with added functionality for training and validation.

    This model takes in many more parameters that SimpleModel, allowing us to
    experiement with many hyperparameter and architecture choices in order to decide
    what is best for our task & data

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float = 0.001,
        hidden_layer_num: int = 1,
        hidden_layer_sizes: list = [128],
        activation: str = "ReLU",  # can be "ReLU", "Sigmoid", "Tanh", "LeakyRelU"
        optimizer: str = "Adam",  # can be "Adam", "SGD", "RMSprop"
        L2_regularization_term: float = 0.0,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Constructor of CustomizableModel.

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
        :param hidden_layer_num: The number of hidden layers in the model
        :type hidden_layer_num: int
        :param hidden_layer_sizes: The size of each hidden layer in the model
        :type hidden_layer_sizes: list

        """
        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be an integer")
        if not isinstance(output_dim, int):
            raise TypeError("output_dim must be an integer")
        if not isinstance(lr, float) or lr <= 0:
            raise TypeError("lr must be a positive float")
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive integers")
        if not isinstance(hidden_layer_num, int):
            raise TypeError("hidden_layer_num must be an integer")
        if not isinstance(hidden_layer_sizes, list) or not all(
            isinstance(i, int) for i in hidden_layer_sizes
        ):
            raise TypeError("hidden_layer_sizes must be a list of integers")
        if len(hidden_layer_sizes) != hidden_layer_num:
            raise ValueError(
                "hidden_layer_sizes must have length equal to hidden_layer_num"
            )
        if not isinstance(activation, str) or activation not in [
            "ReLU",
            "Sigmoid",
            "Tanh",
            "LeakyReLU",
        ]:
            raise ValueError(
                "activation must be one of 'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU'"
            )
        if not isinstance(optimizer, str) or optimizer not in [
            "Adam",
            "SGD",
            "RMSprop",
        ]:
            raise ValueError("optimizer must be one of 'Adam', 'SGD', 'RMSprop'")
        if not isinstance(L2_regularization_term, float) or L2_regularization_term < 0:
            raise TypeError("L2_regularization_term must be a non-negative float")
        if not isinstance(dropout_rate, float) or dropout_rate < 0 or dropout_rate > 1:
            raise TypeError("dropout_rate must be a float between 0 and 1")

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.hidden_layer_num = hidden_layer_num
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.L2_regularization_term = L2_regularization_term
        self.save_hyperparameters()

        match activation:
            case "ReLU":
                self.activation = nn.ReLU()
            case "Sigmoid":
                self.activation = nn.Sigmoid()
            case "Tanh":
                self.activation = nn.Tanh()
            case "LeakyReLU":
                self.activation = nn.LeakyReLU()
            case _:
                raise ValueError(
                    "activation must be one of 'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU'"
                )

        self.input_layer = nn.Linear(input_dim, hidden_layer_sizes[0])
        self.hidden_layers = nn.ModuleList([])
        for i in range(hidden_layer_num - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            )
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_dim)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model (i.e. how predictions are made for a given input)

        :param x: The input data to the model (minus the target y values)
        :type x: torch.Tensor
        :return: The predicted y values for the input x, this is a tensor of shape
            (batch_size, output_dim)
        :rtype: torch.Tensor

        """
        x = self.dropout(self.activation(self.input_layer(x)))
        for hidden_layer in self.hidden_layers:
            x = self.dropout(self.activation(hidden_layer(x)))
        x = self.output_layer(x)
        return x

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
        return loss

    def configure_optimizers(self) -> Optimizer:
        """
        Configure the optimizer for the model.

        :return: The optimizer for the model
        :rtype: Optimizer

        """
        match self.optimizer:
            case "Adam":
                return torch.optim.Adam(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.L2_regularization_term,
                )
            case "SGD":
                return torch.optim.SGD(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.L2_regularization_term,
                )
            case "RMSprop":
                return torch.optim.RMSprop(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.L2_regularization_term,
                )
            case _:
                raise ValueError("optimizer must be one of 'Adam', 'SGD', 'RMSprop'")
