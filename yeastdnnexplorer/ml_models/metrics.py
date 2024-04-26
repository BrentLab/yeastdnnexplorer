import torch
import torch.nn.functional as F
from torchmetrics import Metric


class SMSE(Metric):
    """
    A class for computing the standardized mean squared error (SMSE) metric.

    This metric is defined as the mean squared error divided by the variance of the true
    values (the target data). Because we are dividing by the variance of the true
    values, this metric is scale-independent and does not depend on the mean of the true
    values. It allows us to effectively compare models drawn from different datasets
    with differring scales or means (as long as their variances are at least relatively
    similar)

    """

    def __init__(self):
        """Initialize the SMSE metric."""
        super().__init__()
        self.add_state("mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("variance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Update the metric with new predictions and true values.

        :param y_pred: The predicted y values
        :type y_pred: torch.Tensor
        :param y_true: The true y values
        :type y_true: torch.Tensor

        """
        self.mse += F.mse_loss(y_pred, y_true, reduction="sum")
        self.variance += torch.var(y_true, unbiased=False) * y_true.size(
            0
        )  # Total variance (TODO should we have unbiased=False here?)
        self.num_samples += y_true.numel()

    def compute(self):
        """
        Compute the SMSE metric.

        :return: The SMSE metric
        :rtype: torch.Tensor

        """
        mean_mse = self.mse / self.num_samples
        mean_variance = self.variance / self.num_samples
        return mean_mse / mean_variance


# TODO move this to be a metric class
def compute_nrmse(self, y_pred, y_true):
    """
    Compute the Normalized Root Mean Squared Error. This can be used to better compare
    models trained on different datasets with differnet scales, although it is not
    perfectly scale invariant.

    :param y_pred: The predicted y values
    :type y_pred: torch.Tensor
    :param y_true: The true y values
    :type y_true: torch.Tensor
    :return: The normalized root mean squared error
    :rtype: torch.Tensor

    """
    rmse = torch.sqrt(F.mse_loss(y_pred, y_true))

    # normalize with the range of true y values
    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range
    return nrmse