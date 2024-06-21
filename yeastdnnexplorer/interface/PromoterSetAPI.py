import os

from yeastdnnexplorer.interface.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class PromoterSetAPI(AbstractRecordsAndFilesAPI):
    """
    Class to interact with the PromoterSetAPI endpoint.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the PromoterSetAPI object.

        :param kwargs: parameters to pass through AbstractRecordsAndFilesAPI
            to AbstractAPI.
        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            ["id", "name"],
        )

        url = kwargs.pop("url", os.getenv("PROMOTERSET_URL", None))

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
