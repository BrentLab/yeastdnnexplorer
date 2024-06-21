import os

from yeastdnnexplorer.interface.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class ExpressionAPI(AbstractRecordsAndFilesAPI):
    """
    Class to interact with the ExpressionAPI endpoint.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the ExpressionAPI object.

        :param kwargs: parameters to pass through AbstractRecordsAndFilesAPI
            to AbstractAPI.
        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "regulator",
                "regulator_locus_tag",
                "regulator_symbol",
                "batch",
                "control",
                "mechanism",
                "restriction",
                "time",
                "source",
                "source_time",
                "lab",
                "assay",
                "workflow",
            ],
        )

        url = kwargs.pop("url", os.getenv("EXPRESSION_URL", None))

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
