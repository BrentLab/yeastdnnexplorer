import os

from yeastdnnexplorer.interface.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class BindingAPI(AbstractRecordsAndFilesAPI):
    """Class to interact with the BindingAPI endpoint."""

    def __init__(self, **kwargs) -> None:
        """
        Initialize the BindingAPI object.

        :param kwargs: parameters to pass through AbstractRecordsAndFilesAPI to
            AbstractAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "regulator",
                "regulator_locus_tag",
                "regulator_symbol",
                "batch",
                "replicate",
                "source",
                "source_orig_id",
                "strain",
                "condition",
                "lab",
                "assay",
                "workflow",
                "data_usable",
            ],
        )

        url = kwargs.pop("url", os.getenv("BINDING_URL", None))

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
