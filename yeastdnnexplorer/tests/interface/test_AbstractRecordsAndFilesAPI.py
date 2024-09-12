import gzip
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import pytest
import responses
from aioresponses import aioresponses

from yeastdnnexplorer.interface.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)

# The following test is commented out because it requires a running server -- this is
# how I retrieved the data for the tests below. The data is saved in the snapshot
# directory
#
# @pytest.mark.asyncio
# async def test_save_response_records_and_files(snapshot):
#     async with aiohttp.ClientSession() as session:
#         url = "http://127.0.0.1:8001/api/promotersetsig/export"
#         async with session.get(
#             url,
#             headers={
#                 "Authorization": f"token {os.getenv('TOKEN')}",
#                 "Content-Type": "application/json",
#             },
#             params={
#                 "regulator_symbol": "HAP5",
#                 "workflow": "nf_core_callingcards_dev",
#                 "data_usable": "pass",
#             },
#         ) as response:
#             response.raise_for_status()
#             response_text = await response.text()
#             snapshot.assert_match(response_text)
#             assert response.status == 200


# @pytest.mark.asyncio
# async def test_save_response_records_and_files():
#     async with aiohttp.ClientSession() as session:
#         url = "http://127.0.0.1:8001/api/promotersetsig/record_table_and_files"
#         async with session.get(
#             url,
#             headers={
#                 "Authorization": f"token {os.getenv('TOKEN')}",
#                 "Content-Type": "application/gzip",
#             },
#             params={
#                 "regulator_symbol": "HAP5",
#                 "workflow": "nf_core_callingcards_dev",
#                 "data_usable": "pass",
#             },
#         ) as response:
#             response.raise_for_status()
#             response_content = await response.read()
#             with open("saved_response.tar.gz", "wb") as f:
#                 f.write(response_content)
#             assert response.status == 200


def promotersetsig_csv_gzip() -> bytes:
    # Define the data as a dictionary
    data = {
        "id": [10690, 10694, 10754, 10929, 10939],
        "uploader_id": [1, 1, 1, 1, 1],
        "upload_date": ["2024-03-26"] * 5,
        "modifier_id": [1, 1, 1, 1, 1],
        "modified_date": [
            "2024-03-26 14:28:43.825628+00:00",
            "2024-03-26 14:28:44.739775+00:00",
            "2024-03-26 14:29:01.837335+00:00",
            "2024-03-26 14:29:45.379790+00:00",
            "2024-03-26 14:29:47.853980+00:00",
        ],
        "binding_id": [4079, 4083, 4143, 4318, 4327],
        "promoter_id": [4, 4, 4, 4, 4],
        "background_id": [6, 6, 6, 6, 6],
        "fileformat_id": [5, 5, 5, 5, 5],
        "file": [
            "promotersetsig/10690.csv.gz",
            "promotersetsig/10694.csv.gz",
            "promotersetsig/10754.csv.gz",
            "promotersetsig/10929.csv.gz",
            "promotersetsig/10939.csv.gz",
        ],
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Convert the DataFrame to CSV and compress it using gzip
    csv_buffer = BytesIO()
    with gzip.GzipFile(fileobj=csv_buffer, mode="w") as gz:
        df.to_csv(gz, index=False)

    # Get the gzipped data as bytes
    return csv_buffer.getvalue()


class ConcreteRecordsAndFilesAPI(AbstractRecordsAndFilesAPI):
    """Concrete implementation of AbstractRecordsAndFilesAPI for testing purposes."""

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        pass

    def update(self, df: Any, **kwargs) -> Any:
        pass

    def delete(self, id: str, **kwargs) -> Any:
        pass

    def submit(self, post_dict: dict, **kwargs) -> Any:
        pass  # Implement for testing if necessary

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        pass  # Implement for testing if necessary


@pytest.fixture
@responses.activate
def api_client():
    valid_url = "http://127.0.0.1:8001/api/promotersetsig"
    responses.add(responses.HEAD, valid_url, status=200)
    return ConcreteRecordsAndFilesAPI(url=valid_url, token="my_token")


@pytest.mark.asyncio
async def test_read_without_files(snapshot, api_client):
    with aioresponses() as m:
        # Mock the HTTP response with the saved snapshot response
        m.get(
            "http://127.0.0.1:8001/api/promotersetsig/export",
            status=200,
            body=promotersetsig_csv_gzip(),
            headers={"Content-Type": "application/gzip"},
        )

        result = await api_client.read()
        assert isinstance(result.get("metadata"), pd.DataFrame)
        assert result.get("metadata").shape == (
            5,
            10,
        )


# chatGPT and I went through many iterations of trying to mock two endpoints at once.
# no success. the retrieve_files is untested outside of the tutorial notebook as a
# result
#
# @pytest.mark.asyncio
# async def test_read_with_responses(snapshot, api_client):
#     with responses.RequestsMock() as rsps:
#         # Mock the /export endpoint
#         rsps.add(
#             responses.GET,
#             "http://127.0.0.1:8001/api/promotersetsig/export",
#             body=promotersetsig_csv_gzip(),
#             status=200,
#             content_type="text/csv",
#         )

#         # Path to the tar.gz file
#         tar_gz_file_path = os.path.join(
#             os.path.dirname(__file__),
#             "snapshots",
#             "promotersetsig_records_and_files.tar.gz",
#         )

#         # Read the content of the tar.gz file
#         with open(tar_gz_file_path, "rb") as tar_gz_file:
#             tar_gz_content = tar_gz_file.read()

#         # Mock the /record_table_and_files endpoint
#         rsps.add(
#             responses.GET,
#             "http://127.0.0.1:8001/api/promotersetsig/record_table_and_files",
#             body=tar_gz_content,
#             status=200,
#             content_type="application/gzip",
#         )

#         # Helper function to create a mock ClientResponse
#         async def create_mock_response(url, method, body, content_type, status):
#             return MockClientResponse(
#                 method, URL(url), status, {"Content-Type": content_type}, body
#             )

#         # Patch aiohttp.ClientSession.get to use our mocked responses
#         async def mock_get(self, url, **kwargs):
#             if "export" in url:
#                 return await create_mock_response(
#                     url,
#                     "GET",
#                     promotersetsig_csv_gzip().encode(),
#                     "text/csv",
#                     200,
#                 )
#             elif "record_table_and_files" in url:
#                 return await create_mock_response(
#                     url,
#                     "GET",
#                     tar_gz_content,
#                     "application/gzip",
#                     200,
#                 )
#             else:
#                 raise ValueError("Unexpected URL")

#         with patch("aiohttp.ClientSession.get", new=mock_get):
#             # Test the read method without retrieving files
#             result = await api_client.read()
#             assert isinstance(result.get("metadata"), pd.DataFrame)
#             assert result.get("metadata").shape == (5, 10)

#             # Test the read method with retrieving files
#             result = await api_client.read(retrieve_files=True)
#             assert isinstance(result.get("metadata"), pd.DataFrame)
#             assert result.get("metadata").shape == (5, 10)
#             assert isinstance(result.get("data"), dict)
#             assert len(result.get("data")) == 5
#             assert all(isinstance(v, pd.DataFrame) \
#                     for v in result.get("data").values())

# test the _detect_delimiter method ####


def test_detect_delimiter_errors(api_client):
    # test that a FileNotFound error is raised if the file does not exist
    with pytest.raises(FileNotFoundError):
        api_client._detect_delimiter("non_existent_file.csv")

    with NamedTemporaryFile(mode="w", suffix=".csv.gz") as tmpfile:
        tmpfile.write("col1,col2,col3\nval1,val2,val3")
        tmpfile.flush()
        tmpfile_path = tmpfile.name

        with pytest.raises(gzip.BadGzipFile):
            api_client._detect_delimiter(tmpfile_path)


def test_comma_delimiter(api_client):
    with NamedTemporaryFile(mode="w", suffix=".csv") as tmpfile:
        tmpfile.write("col1,col2,col3\nval1,val2,val3")
        tmpfile.flush()
        tmpfile_path = tmpfile.name

        delimiter = api_client._detect_delimiter(tmpfile_path)
        assert delimiter == ","


def test_tab_delimiter(api_client):
    with NamedTemporaryFile(mode="w", suffix=".csv") as tmpfile:
        tmpfile.write("col1\tcol2\tcol3\nval1\tval2\tval3")
        tmpfile.flush()
        tmpfile_path = tmpfile.name

        delimiter = api_client._detect_delimiter(tmpfile_path)
        assert delimiter == "\t"


def test_space_delimiter(api_client):
    with NamedTemporaryFile(mode="w", suffix=".csv") as tmpfile:
        tmpfile.write("col1 col2 col3\nval1 val2 val3")
        tmpfile.flush()
        tmpfile_path = tmpfile.name

        delimiter = api_client._detect_delimiter(tmpfile_path)
        assert delimiter == " "


def test_gzipped_file(api_client):
    with NamedTemporaryFile(suffix=".csv.gz") as tmpfile:
        with gzip.open(tmpfile.name, "wt") as gzfile:
            gzfile.write("col1,col2,col3\nval1,val2,val3")
            gzfile.flush()
        tmpfile_path = tmpfile.name

        delimiter = api_client._detect_delimiter(tmpfile_path)
        assert delimiter == ","
