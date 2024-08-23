import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import requests  # type: ignore

from yeastdnnexplorer.interface.Cache import Cache
from yeastdnnexplorer.interface.ParamsDict import ParamsDict


class AbstractAPI(ABC):
    """
    Abstract base class for creating API clients that require token authentication.

    This class provides a template for connecting to a cache for caching API responses,
    validating parameters against a list of valid keys, and provides an interface for
    CRUD operations.

    """

    def __init__(
        self,
        url: str = "",
        token: str = "",
        **kwargs,
    ):
        """
        Initialize the API client.

        :param url: The API endpoint URL. Defaults to the `BASE_URL`
            environment variable.
        :param token: The authentication token. Defaults to the `TOKEN`
            environment variable.
        :param valid_param_keys: A list of valid parameter keys for the API.
        :param params: A ParamsDict object containing parameters for the API request.
        :param cache: a Cache object for caching API responses.
        :param kwargs: Additional keyword arguments that may be passed on to the
            ParamsDict and Cache constructors.

        """
        self.logger = logging.getLogger(__name__)
        self._token = token or os.getenv("TOKEN", "")
        self.url = url or os.getenv("BASE_URL", "")
        self.params = ParamsDict(
            params=kwargs.pop("params", {}),
            valid_keys=kwargs.pop("valid_keys", []),
        )
        self.cache = Cache(
            maxsize=kwargs.pop("maxsize", 100), ttl=kwargs.pop("ttl", 300)
        )

    @property
    def header(self) -> dict[str, str]:
        """The HTTP authorization header."""
        return {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json",
        }

    @property
    def url(self) -> str:
        """The URL for the API."""
        return self._url  # type: ignore

    @url.setter
    def url(self, value: str) -> None:
        if not value:
            self._url = None
        elif hasattr(self, "token") and self.token:
            # validate the URL with the new token
            self._is_valid_url(value)
            self._url = value
        else:
            self.logger.warning("No token provided: URL un-validated")
            self._url = value

    @property
    def token(self) -> str:
        """The authentication token for the API."""
        return self._token

    @token.setter
    def token(self, value: str) -> None:
        self._token = value
        # validate the URL with the new token
        if hasattr(self, "url") and self.url:
            self.logger.info("Validating URL with new token")
            self._is_valid_url(self.url)

    @property
    def cache(self) -> Cache:
        """The cache object for caching API responses."""
        return self._cache

    @cache.setter
    def cache(self, value: Cache) -> None:
        self._cache = value

    @property
    def params(self) -> ParamsDict:
        """The ParamsDict object containing parameters for the API request."""
        return self._params

    @params.setter
    def params(self, value: ParamsDict) -> None:
        self._params = value

    def push_params(self, params: dict[str, Any]) -> None:
        """Adds or updates parameters in the ParamsDict."""
        self.params.update(params)

    def pop_params(self, keys: list[str] | None = None) -> None:
        """Removes parameters from the ParamsDict."""
        if keys is None:
            self.params.clear()
            return
        if keys is not None and not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            del self.params[key]

    @abstractmethod
    def create(self, data: dict[str, Any], **kwargs) -> Any:
        """Placeholder for the create method."""
        raise NotImplementedError(
            f"`create()` is not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def read(self, **kwargs) -> Any:
        """Placeholder for the read method."""
        raise NotImplementedError(
            f"`read()` is not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        """Placeholder for the update method."""
        raise NotImplementedError(
            f"`update()` is not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def delete(self, id: str, **kwargs) -> Any:
        """Placeholder for the delete method."""
        raise NotImplementedError(
            f"`delete()` is not implemented for {self.__class__.__name__}"
        )

    def _is_valid_url(self, url: str) -> None:
        """
        Confirms that the URL is valid and the header authorization is appropriate.

        :param url: The URL to validate.
        :type url: str
        :raises ValueError: If the URL is invalid or the token is not set.

        """
        try:
            response = requests.head(url, headers=self.header, allow_redirects=True)
            if response.status_code != 200:
                raise ValueError("Invalid URL or token provided. Check both.")
        except requests.RequestException as e:
            raise AttributeError(f"Error validating URL: {e}") from e
        except AttributeError as e:
            self.logger.error(f"Error validating URL: {e}")

    def _cache_get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache if configured.

        :param key: The key to retrieve from the cache.
        :type key: str
        :param default: The default value to return if the key is not found.
        :type default: any, optional
        :return: The value from the cache or the default value.
        :rtype: any

        """
        return self.cache.get(key, default=default)

    def _cache_set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache if configured.

        :param key: The key to set in the cache.
        :type key: str
        :param value: The value to set in the cache.
        :type value: any

        """
        self.cache.set(key, value)

    def _cache_list(self) -> list[str]:
        """List keys in the cache if configured."""
        return self.cache.list()

    def _cache_delete(self, key: str) -> None:
        """
        Delete a key from the cache if configured.

        :param key: The key to delete from the cache.
        :type key: str

        """
        self.cache.delete(key)
