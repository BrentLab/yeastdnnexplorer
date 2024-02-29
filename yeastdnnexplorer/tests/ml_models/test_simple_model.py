import pytest
import torch

from yeastdnnexplorer.ml_models.simple_model import SimpleModel


@pytest.fixture
def model():
    return SimpleModel(input_dim=4, output_dim=4)


def test_model_forward_pass(model):
    x = torch.randn(32, 4)  # 32 is batch size, 4 is input dim
    output = model(x)
    assert output.shape == (32, 4)  # 32 is batch size, 4 is output dim


def test_model_training_step(model):
    batch = (torch.randn(32, 4), torch.randn(32, 4))  # 32 is batch size, 4 is input dim
    batch_idx = 0
    loss = model.training_step(batch, batch_idx)
    assert loss.ndim == 0  # loss should be a scalar (0 dimensional tensor)


def test_model_validation_step(model):
    batch = (torch.randn(32, 4), torch.randn(32, 4))  # 32 is batch size, 4 is input dim
    batch_idx = 0
    loss = model.validation_step(batch, batch_idx)
    assert loss.ndim == 0  # loss should be a scalar (0 dimensional tensor)


def test_model_test_step(model):
    batch = (torch.randn(32, 4), torch.randn(32, 4))  # 32 is batch size, 4 is input dim
    batch_idx = 0
    loss = model.test_step(batch, batch_idx)
    assert loss.ndim == 0  # loss should be a scalar (0 dimensional tensor)
