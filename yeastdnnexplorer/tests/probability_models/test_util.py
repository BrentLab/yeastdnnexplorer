import pytest
import torch
from sklearn.metrics import explained_variance_score
from yeastdnnexplorer.probability_models.util import calculate_explained_variance  e


# Create a test model directly
def create_test_model():
    model = torch.nn.Linear(10, 1)
    return model


# Create a specific model with known weights for testing
def create_specific_model():
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(1.0)
    return model


# Create test data directly
def create_test_data():
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    return dataloader


# Create specific test data for testing
# This data uses the linear relationship: y = 2x + 1
# You can modify this if needed to make new tests in the future
def create_specific_data():
    x = torch.arange(0, 10, dtype=torch.float32).unsqueeze(1)
    y = 2 * x + 1  
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    return dataloader

class DataModule:
        def test_dataloader(self):
            return dataloader

def test_calculate_explained_variance():
    # Create test data
    dataloader = create_test_data()

    # Create a test model and set it to evaluation mode to prevent model from udpating
    model = create_test_model()
    model.eval()

    # Create a mock data module structure
    data_module = DataModule()

    # Calculate explained variance using the function
    explained_variance = calculate_explained_variance(model, data_module)

    # Assert that the explained variance is a float and between 0-1 (it could be neg)
    assert isinstance(explained_variance, float)
    assert 0 <= explained_variance <= 1


def test_specific_model_explained_variance():
    # Create specific test data based on the functions at the top
    dataloader = create_specific_data()

    # Create a specific model with known weights and set it to evaluation mode
    model = create_specific_model()
    model.eval()

    # Create a mock data module structure
    data_module = DataModule()

    # Calculate explained variance using the function
    explained_variance = calculate_explained_variance(model, data_module)

    # Expected explained variance is 1 since the model should perfectly fit the line
    expected_explained_variance = 1.0

    # Asset that the calculated explained variance is a float and equal to 1.0
    assert isinstance(explained_variance, float), "Explained variance should be a float"
    assert (
        explained_variance == expected_explained_variance
    ), f"Explained variance should be {expected_explained_variance}"


if __name__ == "__main__":
    pytest.main([__file__])
