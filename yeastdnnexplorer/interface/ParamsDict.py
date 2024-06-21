class ParamsDict(dict):
    """
    A dictionary subclass that ensures all keys are strings and supports
    multiple key-value assignments at once.

    This class is designed to be used for passing parameters to HTTP requests
    and extends the base dictionary class, ensuring that insertion order is preserved.
    """

    def __init__(self, params=None):
        """
        Initialize the ParamsDict with optional initial parameters.

        :param params: A dictionary of initial parameters. All keys must be strings.
        :type params: dict, optional
        :raises ValueError: If `params` is not a dictionary or if any of the keys are
            not strings.
        """
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")
        if len(params) > 0 and not all(isinstance(k, str) for k in params.keys()):
            raise ValueError("params must be a dictionary with string keys")
        super().__init__(params)

    def __setitem__(self, key, value):
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
            super().__setitem__(key, value)
        elif isinstance(key, list) and isinstance(value, list):
            if len(key) != len(value):
                raise ValueError("Length of keys and values must match")
            for k, v in zip(key, value):
                super().__setitem__(k, v)
        else:
            raise KeyError("Key must be a string or list of strings")

    def __getitem__(self, key):
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

    def __delitem__(self, key):
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

    def __repr__(self):
        """
        Return a string representation of the ParamsDict.

        :return: A string representation of the ParamsDict.
        :rtype: str
        """
        return f"ParamsDict({super().__repr__()})"

    def __str__(self):
        """
        Return a human-readable string representation of the ParamsDict.

        :return: A human-readable string representation of the ParamsDict.
        :rtype: str
        """
        return ", ".join(f"{k}: {v}" for k, v in self.items())

    def as_dict(self):
        """
        Convert the ParamsDict to a standard dictionary.

        :return: A standard dictionary with the same items as the ParamsDict.
        :rtype: dict
        """
        return dict(self)
