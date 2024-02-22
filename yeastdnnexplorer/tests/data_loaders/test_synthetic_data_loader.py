import pytest
from torch.utils.data import DataLoader

from yeastdnnexplorer.data_loaders.synthetic_data_loader import SyntheticDataLoader


@pytest.fixture
def data_module():
    return SyntheticDataLoader()


def test_data_loading(data_module):
    data_module.prepare_data()
    data_module.setup()
    assert isinstance(data_module.train_dataloader(), DataLoader)
    assert isinstance(data_module.val_dataloader(), DataLoader)
    assert isinstance(data_module.test_dataloader(), DataLoader)
