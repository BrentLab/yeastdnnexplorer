import os

from yeastdnnexplorer.interface.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class PromoterSetSigAPI(AbstractRecordsAndFilesAPI):
    """Class to interact with the PromoterSetSigAPI endpoint."""

    def __init__(self, **kwargs) -> None:
        """
        Initialize the PromoterSetSigAPI object.

        :param kwargs: parameters to pass through AbstractRecordsAndFilesAPI to
            AbstractAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "binding",
                "promoter",
                "promoter_name",
                "background",
                "background_name",
                "regulator_locus_tag",
                "regulator_symbol",
                "batch",
                "replicate",
                "source",
                "lab",
                "assay",
                "workflow",
                "data_usable",
            ],
        )

        url = kwargs.pop("url", os.getenv("PROMOTERSETSIG_URL", None))

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
