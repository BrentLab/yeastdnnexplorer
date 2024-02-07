import pytorch_lightning as pl
import torch.nn as nn
import torch

class SimpleModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr):
        super(SimpleModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        # define layers for the model here
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear1(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
