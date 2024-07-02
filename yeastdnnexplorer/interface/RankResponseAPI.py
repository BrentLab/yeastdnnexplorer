import gzip
import json
import logging
import os
import tarfile
import tempfile
from collections.abc import Callable
from typing import Any

import aiohttp
import pandas as pd

from yeastdnnexplorer.interface.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class RankResponseAPI(AbstractRecordsAndFilesAPI):
    """
    A class to interact with the Rank Response API.

    Retrieves rank response data from the database.

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the RankResponseAPI object. This will serve as an interface to the
        RankResponse endpoint of both the database and the application cache.

        :param url: The URL of the Rank Response API
        :param kwargs: Additional parameters to pass to AbstractAPI.

        """
        super().__init__(
            url=kwargs.pop("url", os.getenv("PROMOTERSETSIG_URL", "")),
            export_files_url_suffix="rankresponse",
            valid_param_keys=kwargs.pop(
                "valid_param_keys", ["promotersetsig_id", "expression_id"]
            ),
            **kwargs,
        )

    async def read(
        self,
        callback: Callable[
            [pd.DataFrame, dict[str, Any] | None, Any], Any
        ] = lambda metadata, data, cache, **kwargs: (
            {"metadata": metadata, "data": data}
        ),
        retrieve_files: bool = True,
        **kwargs,
    ) -> Any:
        """
        Retrieve data from the Rank Response API.

        :param callback: The function to call with the metadata from the Rank Response
            API.
        :param kwargs: Additional parameters to pass to the callback function.
        :return: The result of the callback function.

        """
        if not callable(callback) or {"metadata", "data", "cache"} - set(
            callback.__code__.co_varnames
        ):
            raise ValueError(
                "The callback must be a callable function with `metadata`, ",
                "`data`, and `cache` as parameters.",
            )

        if not retrieve_files:
            self.logger.warning(
                "The RankResponseAPI does not support "
                "`retrieve_files=False`. Setting it to `True`."
            )
            retrieve_files = True

        required_params = {"promotersetsig_id", "expression_id"}
        if not required_params.issubset(self.params.keys()):
            raise ValueError(
                "`promotersetsig_id` and `expression_id` must be in params"
            )

        additional_args = kwargs

        cache_key = f"{self.params['promotersetsig_id']}_{self.params['expression_id']}"
        cached_result = self._cache_get(cache_key)

        if cached_result is not None:
            return callback(
                cached_result["metadata"],
                cached_result["data"],
                self.cache,
                **additional_args,
            )
        else:
            async with aiohttp.ClientSession() as session:
                rankresponse_url = (
                    f"{self.url.rstrip('/')}/{self.export_files_url_suffix}"
                )
                self.logger.debug("rankresponse_url: %s", rankresponse_url)

                try:
                    async with session.get(
                        rankresponse_url, headers=self.header, params=self.params
                    ) as response:
                        response.raise_for_status()
                        tar_data = await response.read()
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".tar.gz"
                        ) as temp_file:
                            temp_file.write(tar_data)
                            temp_file.flush()
                            temp_file.seek(0)
                            try:
                                metadata, data = self._extract_files(temp_file.name)
                                self._cache_set(
                                    cache_key, {"metadata": metadata, "data": data}
                                )
                            finally:
                                os.unlink(temp_file.name)

                        return callback(metadata, data, self.cache, **additional_args)

                except aiohttp.ClientError as e:
                    logging.error(f"Error in GET request: {e}")
                    raise
                except pd.errors.ParserError as e:
                    logging.error(f"Error reading request content: {e}")
                    raise

    def _extract_files(
        self, tar_path: str
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Extract metadata and associated files from a tarball.

        :param tar_path: The path to the tarball file.
        :return: A tuple of metadata DataFrame and a dictionary of DataFrames for each
            file.

        """
        with tarfile.open(tar_path, mode="r:gz") as tar:
            tar_members = tar.getmembers()
            self.logger.debug(
                "Tar file contains: %s", [member.name for member in tar_members]
            )

            # Extract metadata.json
            metadata_member = next(
                (m for m in tar_members if m.name == "metadata.json"), None
            )
            if metadata_member is None:
                raise FileNotFoundError("metadata.json not found in tar archive")

            with tar.extractfile(metadata_member) as f:  # type: ignore
                metadata_dict = json.load(f)
                metadata_df = pd.DataFrame(metadata_dict.values())
                metadata_df["expression_id"] = metadata_dict.keys()

            # Extract CSV files
            data = {}
            for rr_id in metadata_df.id:
                csv_filename = f"{rr_id}.csv.gz"
                member = next((m for m in tar_members if m.name == csv_filename), None)
                if member is None:
                    raise FileNotFoundError(f"{csv_filename} not found in tar archive")

                with tar.extractfile(member) as f:  # type: ignore
                    # Decompress the gzip file before reading it with pandas
                    with gzip.open(f) as gz:
                        data[rr_id] = pd.read_csv(gz)

        return metadata_df, data

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
