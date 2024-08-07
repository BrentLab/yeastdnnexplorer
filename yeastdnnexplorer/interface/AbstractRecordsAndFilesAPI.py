import csv
import gzip
import logging
import os
import tarfile
import tempfile
from collections.abc import Callable
from io import BytesIO
from typing import Any

import aiohttp
import pandas as pd

from yeastdnnexplorer.interface.AbstractAPI import AbstractAPI


class AbstractRecordsAndFilesAPI(AbstractAPI):
    """
    Abstract class to interact with both the records and the data stored in the `file`
    field.

    The return for this class must be records, against the `/export`
    endpoint when `retrieve_files` is False. When `retrieve_files` is True, the cache
    should be checked first. If the file doesn't exist there, it should be retrieved
    from the database against the `/record_table_and_files` endpoint. The file should
    be a tarball with the metadata.csv and the file associated with the record,
    where the file is named according to the `id` field in metadata.csv. Data files
    should be `.csv.gz`.

    """

    def __init__(self, **kwargs):
        """
        Initialize the AbstractRecordsAndFilesAPI object. This will serve as an
        interface to an endpoint that can serve both records and files, and cache the
        file/retrieve from the cache if it exists.

        :param kwargs: parameters to pass to AbstractAPI.

        """
        self.export_url_suffix = kwargs.pop("export_url_suffix", "export")
        self.export_files_url_suffix = kwargs.pop(
            "export_files_url_suffix", "record_table_and_files"
        )
        super().__init__(**kwargs)

    @property
    def export_url_suffix(self) -> str:
        """The URL suffix for exporting records."""
        return self._export_url_suffix

    @export_url_suffix.setter
    def export_url_suffix(self, value: str) -> None:
        self._export_url_suffix = value

    @property
    def export_files_url_suffix(self) -> str:
        """The URL suffix for exporting files."""
        return self._export_files_url_suffix

    @export_files_url_suffix.setter
    def export_files_url_suffix(self, value: str) -> None:
        self._export_files_url_suffix = value

    def _detect_delimiter(self, file_path: str, sample_size: int = 1024) -> str:
        """
        Detect the delimiter of a CSV file.

        :param file_path: The path to the CSV file.
        :type file_path: str
        :param sample_size: The number of bytes to read from the file to detect the
            delimiter. Defaults to 1024.
        :type sample_size: int

        :return: The delimiter of the CSV file.
        :rtype: str

        :raises FileNotFoundError: If the file does not exist.
        :raises gzip.BadGzipFile: If the file is not a valid gzip file.
        """
        try:
            file = (
                gzip.open(file_path, "rt")
                if file_path.endswith(".gz")
                else open(file_path, "rt")
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File {file_path} not found.") from exc

        sample = file.read(sample_size)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        delimiter = dialect.delimiter

        file.close()

        return delimiter

    async def read(
        self,
        callback: Callable[
            [pd.DataFrame, dict[str, Any] | None, Any], Any
        ] = lambda metadata, data, cache, **kwargs: (
            {"metadata": metadata, "data": data}
        ),
        retrieve_files: bool = False,
        **kwargs,
    ) -> Any:
        """
        Retrieve data from the endpoint according to the `retrieve_files` parameter. If
        `retrieve_files` is False, the records will be returned as a dataframe. If
        `retrieve_files` is True, the files associated with the records will be
        retrieved either from the local cache or from the database.

        :param callback: The function to call with the metadata. Signature must
            include `metadata`, `data`, and `cache`.
        :type callback: Callable[[pd.DataFrame, dict[str, Any] | None, Any], Any]
        :param retrieve_files: Boolean. Whether to retrieve the files associated with
            the records. Defaults to False.
        :type retrieve_files: bool
        :param kwargs: Additional arguments to pass to the callback function.
        :type kwargs: Any

        :return: The result of the callback function.
        :rtype: Any

        """
        if not callable(callback) or {"metadata", "data", "cache"} - set(
            callback.__code__.co_varnames
        ):
            raise ValueError(
                "The callback must be a callable function with `metadata`, `data`, ",
                "and `cache` as parameters.",
            )

        export_url = f"{self.url.rstrip('/')}/{self.export_url_suffix}"
        self.logger.debug("export_url: %s", export_url)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    export_url, headers=self.header, params=self.params
                ) as response:
                    response.raise_for_status()
                    content = await response.content.read()
                    with gzip.GzipFile(fileobj=BytesIO(content)) as f:
                        records_df = pd.read_csv(f)

                    if not retrieve_files:
                        return callback(records_df, None, self.cache, **kwargs)
                    else:
                        data_list = await self._retrieve_files(session, records_df)
                        return callback(
                            records_df,
                            data_list,
                            self.cache,
                            **kwargs,
                        )

            except aiohttp.ClientError as e:
                logging.error(f"Error in GET request: {e}")
                raise
            except pd.errors.ParserError as e:
                logging.error(f"Error reading request content: {e}")
                raise

    async def _retrieve_files(
        self, session: aiohttp.ClientSession, records_df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """
        Retrieve files associated with the records either from the local cache or from
        the database.

        :param session: The aiohttp ClientSession.
        :type session: aiohttp.ClientSession
        :param records_df: The DataFrame containing the records.
        :type records_df: pd.DataFrame

        :return: A dictionary where the keys are record IDs and the values are
            DataFrames of the associated files.
        :rtype: dict[str, pd.DataFrame]

        """
        data_list = {}
        for record_id in records_df["id"]:
            data_list[str(record_id)] = await self._retrieve_file(session, record_id)
        return data_list

    async def _retrieve_file(
        self, session: aiohttp.ClientSession, record_id: int
    ) -> pd.DataFrame:
        """
        Retrieve a file associated with a record either from the local cache or from the
        database.

        :param session: The aiohttp ClientSession.
        :type session: aiohttp.ClientSession
        :param record_id: The ID of the record.
        :type record_id: int

        :return: A DataFrame containing the file's data.
        :rtype: pd.DataFrame

        :raises FileNotFoundError: If the file is not found in the tar archive.
        :raises ValueError: If the delimiter is not supported.
        """
        export_files_url = f"{self.url.rstrip('/')}/{self.export_files_url_suffix}"
        self.logger.debug("export_url: %s", export_files_url)
        # Try to get the data from the cache first
        cache_key = str(record_id)
        cached_data = self._cache_get(cache_key)
        if cached_data is not None:
            logging.info(f"Record ID {record_id} retrieved from cache.")
            return pd.read_json(BytesIO(cached_data.encode()))

        # Retrieve from the database if not in cache
        logging.info(
            f"Record ID {record_id} not found in cache. Retrieving from the database."
        )
        try:
            header = self.header.copy()
            header["Content-Type"] = "application/gzip"
            async with session.get(
                export_files_url, headers=header, params={"id": record_id}, timeout=120
            ) as response:
                response.raise_for_status()
                tar_data = await response.read()

            # Create a temporary file for the tarball
            tar_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz")
            try:
                tar_file.write(tar_data)
                tar_file.flush()
                tar_file.seek(0)

                # Create a temporary directory for extraction
                with tempfile.TemporaryDirectory() as extract_dir:
                    # Open the tar file and log its contents
                    with tarfile.open(fileobj=tar_file, mode="r:gz") as tar:
                        tar_members = tar.getmembers()
                        self.logger.debug(
                            "Tar file contains: ",
                            "{[member.name for member in tar_members]}",
                        )

                        # Find the specific file to extract
                        csv_filename = f"{record_id}.csv.gz"
                        member = next(
                            (m for m in tar_members if m.name == csv_filename), None
                        )
                        if member is None:
                            raise FileNotFoundError(
                                f"{csv_filename} not found in tar archive"
                            )

                        # Extract only the specific member
                        tar.extract(member, path=extract_dir)

                    # Read the extracted CSV file
                    csv_path = os.path.join(extract_dir, csv_filename)
                    self.logger.debug(f"Extracted file: {csv_path}")

                    delimiter = self._detect_delimiter(csv_path)

                    # raise an error if the delimiter is not a "," or a "\t"
                    if delimiter not in [",", "\t"]:
                        raise ValueError(
                            f"Delimiter {delimiter} is not supported. Supported delimiters are ',' and '\\t'."
                        )

                    df = pd.read_csv(csv_path, delimiter=delimiter)

                    # Store the data in the cache
                    self._cache_set(cache_key, df.to_json())
            finally:
                os.unlink(tar_file.name)

            return df
        except Exception as e:
            logging.error(f"Error retrieving file for record ID {record_id}: {e}")
            raise
