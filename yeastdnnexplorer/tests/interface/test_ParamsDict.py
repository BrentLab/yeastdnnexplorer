import pytest
import requests  # type: ignore
import responses

from yeastdnnexplorer.interface.ParamsDict import ParamsDict


def test_initialization():
    params = ParamsDict({"b": 2, "a": 1}, valid_keys=["a", "b"])
    assert params == {"a": 1, "b": 2}


def test_getitem():
    params = ParamsDict({"a": 1, "b": 2}, valid_keys=["a", "b"])
    assert params["a"] == 1
    assert params[["a", "b"]] == ParamsDict({"a": 1, "b": 2})
    with pytest.raises(KeyError):
        _ = params["123"]  # Changed from 123 to '123'


def test_setitem():
    params = ParamsDict({"a": 1}, valid_keys=["a", "b", "c", "d"])
    params["b"] = 2
    assert params == {"a": 1, "b": 2}

    params[["c", "d"]] = [3, 4]
    assert params == {"a": 1, "b": 2, "c": 3, "d": 4}

    with pytest.raises(ValueError):
        params[["e", "f"]] = [5]

    with pytest.raises(KeyError):
        params[123] = 5  # type: ignore


def test_delitem():
    params = ParamsDict({"a": 1, "b": 2}, valid_keys=["a", "b"])
    del params["a"]
    assert params == {"b": 2}
    with pytest.raises(KeyError):
        del params["123"]  # Changed from 123 to '123'


def test_repr():
    params = ParamsDict({"a": 1, "b": 2}, valid_keys=["a", "b"])
    assert repr(params) == "ParamsDict({'a': 1, 'b': 2})"


def test_str():
    params = ParamsDict({"a": 1, "b": 2}, valid_keys=["a", "b"])
    assert str(params) == "a: 1, b: 2"


def test_len():
    params = ParamsDict({"a": 1, "b": 2}, valid_keys=["a", "b", "c"])
    assert len(params) == 2
    params["c"] = 3
    assert len(params) == 3


def test_keys_values_items():
    params = ParamsDict({"a": 1, "b": 2}, valid_keys=["a", "b"])
    assert set(params.keys()) == {"a", "b"}
    assert set(params.values()) == {1, 2}
    assert set(params.items()) == {("a", 1), ("b", 2)}


def test_clear():
    params = ParamsDict({"a": 1, "b": 2}, valid_keys=["a", "b"])
    params.clear()
    assert len(params) == 0


def test_as_dict():
    params = ParamsDict({"a": 1, "b": 2}, valid_keys=["a", "b"])
    assert params.as_dict() == {"a": 1, "b": 2}


@responses.activate
def test_requests_integration():
    params = ParamsDict({"a": 1, "b": 2}, valid_keys=["a", "b"])

    url = "https://httpbin.org/get"
    responses.add(responses.GET, url, json={"args": {"a": "1", "b": "2"}}, status=200)

    response = requests.get(url, params=params)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["args"] == {"a": "1", "b": "2"}


if __name__ == "__main__":
    pytest.main()
