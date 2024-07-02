import logging
from collections.abc import Callable
from io import StringIO
from typing import Any

import aiohttp
import pandas as pd

from yeastdnnexplorer.interface.AbstractAPI import AbstractAPI


class AbstractRecordsOnlyAPI(AbstractAPI):
    """Abstract class for CRUD operations on records-only (no file storage)
    endpoints."""

    def __init__(self, **kwargs):
        """
        Initialize the RecordsOnlyAPI object.

        :param kwargs: Additional parameters to pass to AbstractAPI.

        """
        self.logger = logging.getLogger(__name__)
        super().__init__(**kwargs)

    async def read(
        self,
        callback: Callable[
            [pd.DataFrame, dict[str, Any] | None, Any], Any
        ] = lambda metadata, data, cache, **kwargs: {
            "metadata": metadata,
            "data": data,
        },
        export_url_suffix="export",
        **kwargs,
    ) -> Any:
        """
        Retrieve data from the endpoint. The data will be returned as a dataframe. The
        callback function must take metadata, data, and cache as parameters.

        :param callback: The function to call with the data. Signature must
            include `metadata`, `data`, and `cache` as parameters.
        :param export_url_suffix: The URL suffix for the export endpoint. This will
            return a response object with a csv file.
        :param kwargs: Additional arguments to pass to the callback function.
        :return: The result of the callback function.

        """
        if not callable(callback) or {"metadata", "data", "cache"} - set(
            callback.__code__.co_varnames
        ):
            raise ValueError(
                "The callback must be a callable function with `metadata`,",
                "`data`, and `cache` as parameters.",
            )

        export_url = f"{self.url.rstrip('/')}/{export_url_suffix}"
        self.logger.debug("export_url: %s", export_url)

        async with aiohttp.ClientSession() as session:
            try:
                # note that the url and the export suffix are joined such that
                # the url is stripped of any trailing slashes and the export suffix is
                # added without a leading slash
                async with session.get(
                    export_url,
                    headers=self.header,
                    params=self.params,
                ) as response:
                    response.raise_for_status()
                    text = await response.text()
                    records_df = pd.read_csv(StringIO(text))
                    return callback(records_df, None, self.cache, **kwargs)
            except aiohttp.ClientError as e:
                self.logger.error(f"Error in GET request: {e}")
                raise
            except pd.errors.ParserError as e:
                self.logger.error(f"Error reading request content: {e}")
                raise
