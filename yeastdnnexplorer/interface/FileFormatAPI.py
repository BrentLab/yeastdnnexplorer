import os

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class FileFormatAPI(AbstractRecordsOnlyAPI):
    """
    A class to interact with the FileFormatAPI endpoint.
    """

    def __init__(self, **kwargs):
        """
        Initialize the FileFormatAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.
        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "fileformat",
                "fields",
                "separator",
                "feature_identifier_col",
                "effect_col",
                "default_effect_threshold",
                "pval_col",
                "default_pvalue_threshold",
            ],
        )

        url = kwargs.pop("url", os.getenv("FILEFORMAT_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`FILEFORMAT_URL` must be set",
            )

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
