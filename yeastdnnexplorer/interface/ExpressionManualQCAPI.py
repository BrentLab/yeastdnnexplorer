import os

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class ExpressionManualQCAPI(AbstractRecordsOnlyAPI):
    """A class to interact with the ExpressionManualQCAPI endpoint."""

    def __init__(self, **kwargs):
        """
        Initialize the ExpressionManualQCAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "expression",
                "strain_verified",
                "regulator_locus_tag",
                "regulator_symbol",
                "batch",
                "replicate",
                "control",
                "mechanism",
                "restriction",
                "time",
                "source",
                "lab",
                "assay",
                "workflow",
            ],
        )

        url = kwargs.pop("url", os.getenv("EXPRESSIONMANUALQC_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`EXPRESSIONMANUALQC_URL` must be set",
            )

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
