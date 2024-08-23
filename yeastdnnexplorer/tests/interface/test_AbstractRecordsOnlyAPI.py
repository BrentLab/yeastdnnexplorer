import gzip
from typing import Any

import pandas as pd
import pytest
import responses
from aioresponses import aioresponses

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class ConcreteAPI(AbstractRecordsOnlyAPI):
    """Concrete implementation of AbstractRecordsOnlyAPI for testing purposes."""

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        pass  # Implement for testing if necessary

    def update(self, df: Any, **kwargs) -> Any:
        pass  # Implement for testing if necessary

    def delete(self, id: str, **kwargs) -> Any:
        pass  # Implement for testing if necessary


@pytest.fixture
@responses.activate
def api_client():
    valid_url = "https://example.com/api/endpoint"
    responses.add(responses.HEAD, valid_url, status=200)
    return ConcreteAPI(url=valid_url, token="my_token")


@pytest.mark.asyncio
async def test_read(snapshot, api_client):
    with aioresponses() as m:
        # Mocking the response
        mocked_csv = (
            "id,uploader_id,upload_date,modifier_id,modified_date,binding_id,promoter_id,background_id,fileformat_id,file\n"  # noqa: E501
            "10690,1,2024-03-26,1,2024-03-26 14:28:43.825628+00:00,4079,4,6,5,promotersetsig/10690.csv.gz\n"  # noqa: E501
            "10694,1,2024-03-26,1,2024-03-26 14:28:44.739775+00:00,4083,4,6,5,promotersetsig/10694.csv.gz\n"  # noqa: E501
            "10754,1,2024-03-26,1,2024-03-26 14:29:01.837335+00:00,4143,4,6,5,promotersetsig/10754.csv.gz\n"  # noqa: E501
            "10929,1,2024-03-26,1,2024-03-26 14:29:45.379790+00:00,4318,4,6,5,promotersetsig/10929.csv.gz\n"  # noqa: E501
            "10939,1,2024-03-26,1,2024-03-26 14:29:47.853980+00:00,4327,4,6,5,promotersetsig/10939.csv.gz"  # noqa: E501
        )

        # Convert to bytes and gzip the content
        gzipped_csv = gzip.compress(mocked_csv.encode("utf-8"))

        m.get(
            "https://example.com/api/endpoint/export",
            status=200,
            body=gzipped_csv,
            headers={"Content-Type": "application/gzip"},
        )

        result = await api_client.read()
        assert isinstance(result, dict)
        assert isinstance(result.get("metadata"), pd.DataFrame)
        assert result.get("metadata").shape == (5, 10)  # type: ignore


if __name__ == "__main__":
    pytest.main()
