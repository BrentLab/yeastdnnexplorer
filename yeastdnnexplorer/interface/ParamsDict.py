from typing import Any, Union


class ParamsDict(dict):
    """
    A dictionary subclass that ensures all keys are strings and supports multiple key-
    value assignments at once, with validation against a list of valid keys.

    This class is designed to be used for passing parameters to HTTP requests and
    extends the base dictionary class, ensuring that insertion order is preserved.

    """

    def __init__(self, params: dict[str, Any] = {}, valid_keys: list[str] = []) -> None:
        """
        Initialize the ParamsDict with optional initial parameters and valid keys.

        :param params: A dictionary of initial parameters. All keys must be strings.
        :type params: dict, optional
        :param valid_keys: A list of valid keys for validation.
        :type valid_keys: list of str, optional
        :raises ValueError: If `params` is not a dictionary or if any of the keys
            are not strings.

        """
        params = params or {}
        valid_keys = valid_keys or []
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")
        if len(params) > 0 and not all(isinstance(k, str) for k in params.keys()):
            raise ValueError("params must be a dictionary with string keys")
        super().__init__(params)
        self._valid_keys = valid_keys

    def __setitem__(self, key: str | list[str], value: Any | list[Any]) -> None:
        """
        Set a parameter value or multiple parameter values.

        :param key: The parameter key or a list of parameter keys.
        :type key: str or list of str
        :param value: The parameter value or a list of parameter values.
        :type value: any or list of any
        :raises ValueError: If the length of `key` and `value` lists do not match.
        :raises KeyError: If `key` is not a string or a list of strings.

        """
        if isinstance(key, str):
            self._validate_key(key)
            super().__setitem__(key, value)
        elif isinstance(key, list) and isinstance(value, list):
            if len(key) != len(value):
                raise ValueError("Length of keys and values must match")
            for k, v in zip(key, value):
                if not isinstance(k, str):
                    raise KeyError("All keys must be strings")
                self._validate_key(k)
                super().__setitem__(k, v)
        else:
            raise KeyError("Key must be a string or list of strings")

    def __getitem__(self, key: str | list[str]) -> Union[Any, "ParamsDict"]:
        """
        Get a parameter value or a new ParamsDict with specified keys.

        :param key: The parameter key or a list of parameter keys.
        :type key: str or list of str
        :return: The parameter value or a new ParamsDict with the specified keys.
        :rtype: any or ParamsDict
        :raises KeyError: If `key` is not a string or a list of strings.

        """
        if isinstance(key, str):
            return super().__getitem__(key)
        elif isinstance(key, list):
            return ParamsDict({k: dict.__getitem__(self, k) for k in key if k in self})
        else:
            raise KeyError("Key must be a string or list of strings")

    def __delitem__(self, key: str) -> None:
        """
        Delete a parameter by key.

        :param key: The parameter key.
        :type key: str
        :raises KeyError: If `key` is not a string.

        """
        if isinstance(key, str):
            super().__delitem__(key)
        else:
            raise KeyError("Key must be a string")

    def __repr__(self) -> str:
        """
        Return a string representation of the ParamsDict.

        :return: A string representation of the ParamsDict.
        :rtype: str

        """
        return f"ParamsDict({super().__repr__()})"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the ParamsDict.

        :return: A human-readable string representation of the ParamsDict.
        :rtype: str

        """
        return ", ".join(f"{k}: {v}" for k, v in self.items())

    def as_dict(self) -> dict:
        """
        Convert the ParamsDict to a standard dictionary.

        :return: A standard dictionary with the same items as the ParamsDict.
        :rtype: dict

        """
        return dict(self)

    def _validate_key(self, key: str) -> None:
        """Validate that the key is in the list of valid keys."""
        if self._valid_keys and key not in self._valid_keys:
            raise ValueError(f"Invalid parameter key provided: {key}")

    @property
    def valid_keys(self) -> list[str]:
        """Get the list of valid keys."""
        return self._valid_keys

    @valid_keys.setter
    def valid_keys(self, keys: list[str]) -> None:
        """Set the list of valid keys."""
        if not all(isinstance(k, str) for k in keys):
            raise ValueError("valid_keys must be a list of strings")
        self._valid_keys = keys
