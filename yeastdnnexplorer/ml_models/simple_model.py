import pytorch_lightning as pl
import torch.nn as nn
import torch

class SimpleModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr=0.001):
        super(SimpleModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.save_hyperparameters() # needed so that the hyperparameters are saved with the model and can be loaded later (checkpoints wouldn't load without this line)

        # define layers for the model here
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear1(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = nn.functional.mse_loss(y_pred, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
