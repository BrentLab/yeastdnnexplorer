import logging
from collections.abc import Callable

import torch

from yeastdnnexplorer.probability_models.relation_classes import Relation

logger = logging.getLogger(__name__)

from sklearn.metrics import explained_variance_score


def calculate_explained_variance(model, data_module):
    """
    Calculates the explained variance score of the model's predictions on the test
    dataset

    Parameters:
    - model: The trained model to predict with
    - data_module: The data module for the test data loader

    Returns:
    - explained_variance: should be a float between 0 and 1 (but could be negative)
    """

    predictions = []
    targets = []

    # Set the model to evaluation mode to disable dropout,
    # batch normalization, etc.
    model.eval()

    # Disable gradient calculation to save memory and computation
    # Iterate over the test data batches, get the input features (x) and true targets
    # (y) from the batch, make predictions using the model, add everything to both lists

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            x, y = batch
            outputs = model(x)
            predictions.append(outputs)
            targets.append(y)

    # Concatenate all batch predictions and targets into single tensors
    # They should be numpy arrays in order for explained_variance_score to work properly
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    targets = torch.cat(targets, dim=0).cpu().numpy()

    # Calculate and return the explained variance score
    return explained_variance_score(targets, predictions)
