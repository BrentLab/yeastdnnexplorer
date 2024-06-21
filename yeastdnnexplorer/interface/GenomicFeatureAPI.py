import os

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class GenomicFeatureAPI(AbstractRecordsOnlyAPI):
    """
    A class to interact with the GenomicFeatureAPI endpoint.
    """

    def __init__(self, **kwargs):
        """
        Initialize the GenomicFeatureAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.
        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "chr",
                "start",
                "end",
                "strand",
                "type",
                "locus_tag",
                "symbol",
                "source",
                "alias",
                "note",
            ],
        )

        url = kwargs.pop("url", os.getenv("GENOMICFEATURE_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`GENOMICFEATURE_URL` must be set",
            )

        super().__init__(url=url, valid_param_keys=valid_param_keys, **kwargs)

    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass
