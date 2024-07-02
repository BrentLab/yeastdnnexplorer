import os

from yeastdnnexplorer.interface.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class CallingCardsBackgroundAPI(AbstractRecordsAndFilesAPI):
    """Class to interact with the CallingCardsBackgroundAPI endpoint."""

    def __init__(self, **kwargs) -> None:
        """
        Initialize the CallingCardsBackgroundAPI object.

        :param kwargs: parameters to pass through AbstractRecordsAndFilesAPI to
            AbstractAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            ["id", "name"],
        )

        url = kwargs.pop("url", os.getenv("CALLINGCARDSBACKGROUND_URL", None))

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
