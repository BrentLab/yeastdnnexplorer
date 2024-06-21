import os

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class RegulatorAPI(AbstractRecordsOnlyAPI):
    """
    A class to interact with the RegulatorAPI endpoint.
    """

    def __init__(self, **kwargs):
        """
        Initialize the RegulatorAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.
        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "regulator_locus_tag",
                "regulator_symbol",
                "under_development",
            ],
        )

        url = kwargs.pop("url", os.getenv("REGULATOR_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`REGULATOR_URL` must be set",
            )

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
