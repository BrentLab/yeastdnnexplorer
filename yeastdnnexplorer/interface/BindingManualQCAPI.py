import os

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class BindingManualQCAPI(AbstractRecordsOnlyAPI):
    """
    A class to interact with the BindingManualQCAPI endpoint.
    """

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

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
