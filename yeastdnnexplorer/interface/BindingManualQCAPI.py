import os
from typing import Any

import pandas as pd
import requests  # type: ignore

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class BindingManualQCAPI(AbstractRecordsOnlyAPI):
    """A class to interact with the BindingManualQCAPI endpoint."""

    def __init__(self, **kwargs):
        """
        Initialize the BindingManualQCAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "binding",
                "best_datatype",
                "data_usable",
                "passing_replicate",
                "rank_recall",
                "regulator",
                "regulator_locus_tag",
                "regulator_symbol",
                "batch",
                "source",
            ],
        )

        url = kwargs.pop("url", os.getenv("BINDINGMANUALQC_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`BINDINGMANUALQC_URL` must be set",
            )

        self.bulk_update_url_suffix = kwargs.pop(
            "bulk_update_url_suffix", "bulk-update"
        )

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    @property
    def bulk_update_url_suffix(self) -> str:
        """The URL suffix for updating multiple records in the same request."""
        return self._bulk_update_url_suffix

    @bulk_update_url_suffix.setter
    def bulk_update_url_suffix(self, value: str) -> None:
        self._bulk_update_url_suffix = value

    def create(self):
        pass

    def update(self, df: pd.DataFrame, **kwargs: Any) -> requests.Response:
        """
        Update the records in the database.

        :param df: The DataFrame containing the records to update.
        :type df: pd.DataFrame
        :param kwargs: Additional fields to include in the payload.
        :type kwargs: Any
        :return: The response from the POST request.
        :rtype: requests.Response
        :raises requests.RequestException: If the request fails.

        """
        bulk_update_url = (
            f"{self.url.rstrip('/')}/{self.bulk_update_url_suffix.rstrip('/')}/"
        )

        self.logger.debug("bulk_update_url: %s", bulk_update_url)

        # Include additional fields in the payload if provided
        payload = {"data": df.to_dict(orient="records")}
        payload.update(kwargs)

        try:
            response = requests.post(
                bulk_update_url,
                headers=self.header,
                json=payload,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"Error in POST request: {e}")
            raise

    def delete(self):
        pass
