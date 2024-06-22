from typing import Any

import pytest
import responses

from yeastdnnexplorer.interface.AbstractAPI import AbstractAPI
from yeastdnnexplorer.interface.ParamsDict import ParamsDict


class ConcreteAPI(AbstractAPI):
    """Concrete implementation of AbstractAPI for testing purposes."""

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        pass  # Implement for testing if necessary

    def read(self, **kwargs) -> dict[str, Any]:
        return {"id": id}  # Mock implementation for testing

    def update(self, **kwargs) -> Any:
        pass  # Implement for testing if necessary

    def delete(self, id: str, **kwargs) -> Any:
        pass  # Implement for testing if necessary


@pytest.fixture
@responses.activate
def api_client():
    valid_url = "https://valid.url"
    responses.add(responses.HEAD, valid_url, status=200)
    return ConcreteAPI(url=valid_url, token="token")


def test_initialize(snapshot, api_client):
    assert api_client.url == "https://valid.url"
    assert api_client.token == "token"
    assert isinstance(api_client.params, ParamsDict)


def test_push_params(snapshot, api_client):
    params = {"param1": "value1", "param2": "value2"}
    api_client.push_params(params)
    snapshot.assert_match(api_client.params.as_dict(), "push_params")


def test_pop_params(snapshot, api_client):
    params = {"param1": "value1", "param2": "value2"}
    api_client.push_params(params)
    api_client.pop_params(["param1"])
    snapshot.assert_match(api_client.params.as_dict(), "pop_params_after_one_removed")
    api_client.pop_params()
    snapshot.assert_match(api_client.params.as_dict(), "pop_params_after_all_removed")


def test_is_valid_url(snapshot, api_client):
    invalid_url = "https://invalid.url"

    responses.add(responses.HEAD, invalid_url, status=404)

    with pytest.raises(AttributeError):
        api_client.url = invalid_url


def test_cache_operations(snapshot, api_client):
    key = "test_key"
    value = "test_value"

    api_client._cache_set(key, value)
    snapshot.assert_match(api_client._cache_get(key), "cache_get_after_set")

    keys = api_client._cache_list()
    snapshot.assert_match(keys, "cache_list")

    api_client._cache_delete(key)
    snapshot.assert_match(api_client._cache_get(key), "cache_get_after_delete")


if __name__ == "__main__":
    pytest.main()
