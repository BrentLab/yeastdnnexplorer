import os

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class DataSourceAPI(AbstractRecordsOnlyAPI):
    """
    A class to interact with the DataSourceAPI endpoint.
    """

    def __init__(self, **kwargs):
        """
        Initialize the DataSourceAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.
        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            ["id", "fileformat_id", "fileformat", "lab", "assay", "workflow"],
        )

        url = kwargs.pop("url", os.getenv("DATASOURCE_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`DATASOURCE_URL` must be set",
            )

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
