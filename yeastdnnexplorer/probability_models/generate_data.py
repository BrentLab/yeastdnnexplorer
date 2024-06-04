import inspect
import logging
from collections.abc import Callable

import torch

from yeastdnnexplorer.probability_models.relation_classes import Relation

logger = logging.getLogger(__name__)


class GenePopulation:
    """A simple class to hold a tensor boolean 1D vector where 0 is meant to identify
    genes which are unaffected by a given TF and 1 is meant to identify genes which are
    affected by a given TF."""

    def __init__(self, labels: torch.Tensor) -> None:
        """
        Constructor of GenePopulation.

        :param labels: This can be any 1D tensor of boolean values. But it is meant to
            be the output of `generate_gene_population()`
        :type labels: torch.Tensor
        :raises TypeError: If labels is not a tensor
        :raises ValueError: If labels is not a 1D tensor
        :raises TypeError: If labels is not a boolean tensor

        """
        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a tensor")
        if not labels.ndim == 1:
            raise ValueError("labels must be a 1D tensor")
        if not labels.dtype == torch.bool:
            raise TypeError("labels must be a boolean tensor")
        self.labels = labels

    def __repr__(self):
        return f"<GenePopulation size={len(self.labels)}>"


def generate_gene_population(
    total: int = 1000, bound_group: float = 0.3
) -> GenePopulation:
    """
    Generate two sets of genes, one of which will be considered genes which show a
    bound, and the other which does not. The return is a one dimensional boolean tensor
    where a value of '0' means that the gene at that index is part of the unbound group
    and a '1' means the gene at that index is part of the bound group. The length of the
    tensor is the number of genes in this simulated organism.

    :param total: The total number of genes. defaults to 1000
    :type total: int, optional
    :param bound_group: The proportion of genes in the bound group. defaults to 0.3
    :type bound_group: float, optional
    :return: A one dimensional tensor of boolean values where the set of indices with a
        value of '1' are the bound group and the set of indices with a value of '0' are
        the unbound group.
    :rtype: GenePopulation
    :raises TypeError: if total is not an integer
    :raises ValueError: If bound_group is not between 0 and 1

    """
    if not isinstance(total, int):
        raise TypeError("total must be an integer")
    if not 0 <= bound_group <= 1:
        raise ValueError("bound_group must be between 0 and 1")

    bound_group_size = int(total * bound_group)
    logger.info("Generating %s genes with bound", bound_group_size)

    labels = torch.cat(
        (
            torch.ones(bound_group_size, dtype=torch.bool),
            torch.zeros(total - bound_group_size, dtype=torch.bool),
        )
    )[torch.randperm(total)]

    return GenePopulation(labels)


def generate_binding_effects(
    gene_population: GenePopulation,
    background_hops_range: tuple[int, int] = (1, 100),
    unbound_experiment_hops_range: tuple[int, int] = (0, 1),
    bound_experiment_hops_range: tuple[int, int] = (1, 6),
    total_background_hops: int = 1000,
    total_experiment_hops: int = 76,
    pseudocount: float = 1e-10,
) -> torch.Tensor:
    """
    Generate enrichment effects for genes using vectorized operations, based on their
    bound designation, with separate experiment hops ranges for unbound and bound genes.

    Note that the default values are a scaled down version of actual data. See also
    https://github.com/cmatKhan/callingCardsTools/blob/main/callingcardstools/PeakCalling/yeast/enrichment.py

    :param gene_population: A GenePopulation object. See `generate_gene_population()`
    :type gene_population: GenePopulation
    :param background_hops_range: The range of hops for background genes. Defaults to
        (1, 100)
    :type background_hops_range: Tuple[int, int], optional
    :param unbound_experiment_hops_range: The range of hops for unbound genes. Defaults
        to (0, 1)
    :type unbound_experiment_hops_range: Tuple[int, int], optional
    :param bound_experiment_hops_range: The range of hops for bound genes. Defaults to
        (1, 6)
    :type bound_experiment_hops_range: Tuple[int, int], optional
    :param total_background_hops: The total number of background hops. Defaults to 1000
    :type total_background_hops: int, optional
    :param total_experiment_hops: The total number of experiment hops. Defaults to 76
    :type total_experiment_hops: int, optional
    :param pseudocount: A pseudocount to avoid division by zero. Defaults to 1e-10
    :type pseudocount: float, optional
    :return: A tensor of enrichment values for each gene.
    :rtype: torch.Tensor
    :raises TypeError: If gene_population is not a GenePopulation object
    :raises TypeError: If total_background_hops is not an integer
    :raises TypeError: If total_experiment_hops is not an integer
    :raises TypeError: If pseudocount is not a float
    :raises TypeError: If background_hops_range is not a tuple
    :raises TypeError: If unbound_experiment_hops_range is not a tuple
    :raises TypeError: If bound_experiment_hops_range is not a tuple
    :raises ValueError: If background_hops_range is not a tuple of length 2
    :raises ValueError: If unbound_experiment_hops_range is not a tuple of length 2
    :raises ValueError: If bound_experiment_hops_range is not a tuple of length 2

    """
    # NOTE: torch intervals are half open on the right, so we add 1 to the
    # high end of the range to make it inclusive

    # check input
    if not isinstance(gene_population, GenePopulation):
        raise TypeError("gene_population must be a GenePopulation object")
    if not isinstance(total_background_hops, int):
        raise TypeError("total_background_hops must be an integer")
    if not isinstance(total_experiment_hops, int):
        raise TypeError("total_experiment_hops must be an integer")
    if not isinstance(pseudocount, float):
        raise TypeError("pseudocount must be a float")
    for arg, tup in {
        "background_hops_range": background_hops_range,
        "unbound_experiment_hops_range": unbound_experiment_hops_range,
        "bound_experiment_hops_range": bound_experiment_hops_range,
    }.items():
        if not isinstance(tup, tuple):
            raise TypeError(f"{arg} must be a tuple")
        if not len(tup) == 2:
            raise ValueError(f"{arg} must be a tuple of length 2")
        if not all(isinstance(i, int) for i in tup):
            raise TypeError(f"{arg} must be a tuple of integers")

    # Generate background hops for all genes
    background_hops = torch.randint(
        low=background_hops_range[0],
        high=background_hops_range[1] + 1,
        size=(gene_population.labels.shape[0],),
    )

    # Generate experiment hops unbound genes
    unbound_experiment_hops = torch.randint(
        low=unbound_experiment_hops_range[0],
        high=unbound_experiment_hops_range[1] + 1,
        size=(gene_population.labels.shape[0],),
    )
    # Generate experiment hops bound genes
    bound_experiment_hops = torch.randint(
        low=bound_experiment_hops_range[0],
        high=bound_experiment_hops_range[1] + 1,
        size=(gene_population.labels.shape[0],),
    )

    # Use bound designation to select appropriate experiment hops
    experiment_hops = torch.where(
        gene_population.labels == 1, bound_experiment_hops, unbound_experiment_hops
    )

    # Calculate enrichment for all genes
    return (experiment_hops.float() / (total_experiment_hops + pseudocount)) / (
        (background_hops.float() / (total_background_hops + pseudocount)) + pseudocount
    )


def generate_pvalues(
    effects: torch.Tensor,
    large_effect_percentile: float = 0.9,
    large_effect_upper_pval: float = 0.2,
) -> torch.Tensor:
    """
    Generate p-values for genes where larger effects are less likely to be false
    positives.

    :param effects: A tensor of effects
    :type effects: torch.Tensor
    :param large_effect_percentile: The percentile of effects that are considered large
        effects. Defaults to 0.9
    :type large_effect_percentile: float, optional
    :param large_effect_upper_pval: The upper bound of the p-values for large effects.
        Defaults to 0.2
    :return: A tensor of p-values
    :rtype: torch.Tensor
    :raises ValueError: If effects is not a tensor or the values themselves are not
        numeric
    :raises ValueError: If large_effect_percentile is not between 0 and 1
    :raises ValueError: If large_effect_upper_pval is not between 0 and 1

    """
    # check inputs
    if not isinstance(effects, torch.Tensor):
        raise ValueError("effects must be a tensor")
    if not torch.is_floating_point(effects):
        raise ValueError("effects must be numeric")
    if not 0 <= large_effect_percentile <= 1:
        raise ValueError("large_effect_percentile must be between 0 and 1")
    if not 0 <= large_effect_upper_pval <= 1:
        raise ValueError("large_effect_upper_pval must be between 0 and 1")

    # Generate p-values
    pvalues = torch.rand(effects.shape[0])

    # Draw p-values from a uniform distribution where larger abs(effects) are
    # less likely to be false positives
    large_effect_threshold = torch.quantile(torch.abs(effects), large_effect_percentile)
    large_effect_mask = torch.abs(effects) >= large_effect_threshold
    pvalues[large_effect_mask] = (
        torch.rand(torch.sum(large_effect_mask)) * large_effect_upper_pval
    )

    return pvalues


def default_perturbation_effect_adjustment_function(
    binding_enrichment_data: torch.Tensor,
    bound_mean: float,
    unbound_mean: float,
    max_adjustment: float,
    **kwargs,
) -> torch.Tensor:
    """
    Default function to adjust the mean of the perturbation effect based on the
    enrichment score.

    All functions that are passed to generate_perturbation_effects() in the argument
    adjustment_function must have the same signature as this function.

    :param binding_enrichment_data: A tensor of enrichment scores for each gene with
        dimensions [n_genes, n_tfs, 3] where the entries in the third dimension are a
        matrix with columns [label, enrichment, pvalue].
    :type binding_enrichment_data: torch.Tensor
    :param bound_mean: The mean for bound genes.
    :type bound_mean: float
    :param unbound_mean: The mean for unbound genes.
    :type unbound_mean: float
    :param max_adjustment: The maximum adjustment to the base mean based on enrichment.
    :type max_adjustment: float
    :param tf_relationships: Unused in this function. It is only here to match the
        signature of the other adjustment functions.
    :type tf_relationships: dict[int, list[int]], optional
    :return: Adjusted mean as a tensor.
    :rtype: torch.Tensor

    """
    # Extract bound/unbound labels and enrichment scores
    bound_labels = binding_enrichment_data[:, :, 0]
    enrichment_scores = binding_enrichment_data[:, :, 1]

    adjusted_mean_matrix = torch.where(
        bound_labels == 1, enrichment_scores, torch.zeros_like(enrichment_scores)
    )

    for gene_idx in range(bound_labels.shape[0]):
        for tf_index in range(bound_labels.shape[1]):
            if bound_labels[gene_idx, tf_index] == 1:
                # divide its enrichment score by the maximum magnitude possible to
                # create an adjustment multipler that scales with increasing enrichment
                adjustment_multiplier = enrichment_scores[gene_idx, tf_index] / abs(
                    enrichment_scores.max()
                )

                # randomly adjust the gene by some portion of the max adjustment
                adjusted_mean_matrix[gene_idx, tf_index] = bound_mean + (
                    adjustment_multiplier * max_adjustment
                )
            else:
                # related tfs are not all bound, so set the enrichment
                # score to unbound mean
                adjusted_mean_matrix[gene_idx, tf_index] = unbound_mean

    return adjusted_mean_matrix


def perturbation_effect_adjustment_function_with_tf_relationships_boolean_logic(
    binding_enrichment_data: torch.Tensor,
    bound_mean: float,
    unbound_mean: float,
    max_adjustment: float,
    tf_relationships: dict[int, list[Relation]],
) -> torch.Tensor:
    """
    Adjust the mean of the perturbation effect based on the enrichment score and the
    provided binary / boolean or unary relationships between TFs. For each gene, the
    mean of the TF-gene pair's perturbation effect will be adjusted if the TF is bound
    to the gene and all of the Relations associated with the TF are satisfied (ie they
    evaluate to True). These relations could be unary conditions or Ands or Ors between
    TFs. A TF being bound corresponds to a true value, which means And(4, 5) would be
    satisfied is both TF 4 and TF 5 are bound to the gene in question. The adjustment
    will be a random value not exceeding the maximum adjustment.

    :param binding_enrichment_data: A tensor of enrichment scores for each gene with
        dimensions [n_genes, n_tfs, 3] where the entries in the third dimension are a
        matrix with columns [label, enrichment, pvalue].
    :type binding_enrichment_data: torch.Tensor
    :param bound_mean: The mean for bound genes.
    :type bound_mean: float
    :param unbound_mean: The mean for unbound genes.
    :type unbound_mean: float
    :param max_adjustment: The maximum adjustment to the base mean based on enrichment.
    :type max_adjustment: float
    :param tf_relationships: A dictionary where the keys are TF indices and the values
        are lists of Relation objects that represent the conditions that must be met for
        the mean of the perturbation effect associated with the TF-gene pair to be
        adjusted.
    :type tf_relationships: dict[int, list[Relation]]
    :return: Adjusted mean as a tensor.
    :rtype: torch.Tensor
    :raises ValueError: If tf_relationships is not a dictionary between ints and lists
        of Relations
    :raises ValueError: If the tf_relationships dict does not have the same number of
        TFs as the binding_data tensor passed into the function
    :raises ValueError: If the tf_relationships dict has any TFs in the values that are
        not also in the keys or any key or value TFs that are out of bounds for the
        binding_data tensor

    """
    if (
        not isinstance(tf_relationships, dict)
        or not all(isinstance(v, list) for v in tf_relationships.values())
        or not all(isinstance(k, int) for k in tf_relationships.keys())
        or not all(
            isinstance(i, Relation) for v in tf_relationships.values() for i in v
        )
    ):
        raise ValueError(
            "tf_relationships must be a dictionary between \
                ints and lists of Relation objects"
        )
    if not all(
        k in range(binding_enrichment_data.shape[1]) for k in tf_relationships.keys()
    ):
        raise ValueError(
            "all TFs mentioned in tf_relationships must be within \
                the bounds of the binding_data tensor's number of TFs"
        )
    if not len(tf_relationships) == binding_enrichment_data.shape[1]:
        raise ValueError(
            "tf_relationships must have the same number of TFs as \
                the binding_data tensor passed into the function"
        )

    # Extract bound/unbound labels and enrichment scores
    bound_labels = binding_enrichment_data[:, :, 0]  # shape: (num_genes, num_tfs)
    enrichment_scores = binding_enrichment_data[:, :, 1]  # shape: (num_genes, num_tfs)

    # we set all unbound scores to 0, then we will go through and also set any
    # bound scores to unbound_mean if the related boolean statements are not satisfied
    adjusted_mean_matrix = torch.where(
        bound_labels == 1, enrichment_scores, torch.zeros_like(enrichment_scores)
    )  # shape: (num_genes, num_tfs)

    for gene_idx in range(bound_labels.shape[0]):
        for tf_index, relations in tf_relationships.items():
            # check if all relations (boolean relationships)
            # associated with TFs are satisfied
            if bound_labels[gene_idx, tf_index] == 1 and all(
                relation.evaluate(bound_labels[gene_idx].tolist())
                for relation in relations
            ):
                # OLD: adjustment_multiplier = torch.rand(1)
                # divide its enrichment score by the maximum magnitude possible to
                # create an adjustment multipler that scales with increasing enrichment
                adjustment_multiplier = enrichment_scores[gene_idx, tf_index] / abs(
                    enrichment_scores.max()
                )

                # randomly adjust the gene by some portion of the max adjustment
                adjusted_mean_matrix[gene_idx, tf_index] = bound_mean + (
                    adjustment_multiplier * max_adjustment
                )
            else:
                # related tfs are not all bound, set the enrichment score to unbound
                # mean
                adjusted_mean_matrix[gene_idx, tf_index] = unbound_mean

    return adjusted_mean_matrix  # shape (num_genes, num_tfs)


def perturbation_effect_adjustment_function_with_tf_relationships(
    binding_enrichment_data: torch.Tensor,
    bound_mean: float,
    unbound_mean: float,
    max_adjustment: float,
    tf_relationships: dict[int, list[int]],
) -> torch.Tensor:
    """
    Adjust the mean of the perturbation effect based on the enrichment score and the
    provided relationships between TFs. For each gene, the mean of the TF-gene pair's
    perturbation effect will be adjusted if the TF is bound to the gene and all related
    TFs are also bound to the gene. The adjustment will be a random value not exceeding
    the maximum adjustment.

    :param binding_enrichment_data: A tensor of enrichment scores for each gene with
        dimensions [n_genes, n_tfs, 3] where the entries in the third dimension are a
        matrix with columns [label, enrichment, pvalue].
    :type binding_enrichment_data: torch.Tensor
    :param bound_mean: The mean for bound genes.
    :type bound_mean: float
    :param unbound_mean: The mean for unbound genes.
    :type unbound_mean: float
    :param max_adjustment: The maximum adjustment to the base mean based on enrichment.
    :type max_adjustment: float
    :param tf_relationships: A dictionary where the keys are the indices of the TFs and
        the values are lists of indices of other TFs that are related to the key TF.
    :type tf_relationships: dict[int, list[int]]
    :return: Adjusted mean as a tensor.
    :rtype: torch.Tensor
    :raises ValueError: If tf_relationships is not a dictionary between ints and lists
        of ints
    :raises ValueError: If the tf_relationships dict does not have the same number of
        TFs as the binding_data tensor passed into the function
    :raises ValueError: If the tf_relationships dict has any TFs in the values that are
        not also in the keys or any key or value TFs that are out of bounds for the
        binding_data tensor

    """
    if (
        not isinstance(tf_relationships, dict)
        or not all(isinstance(v, list) for v in tf_relationships.values())
        or not all(isinstance(k, int) for k in tf_relationships.keys())
        or not all(isinstance(i, int) for v in tf_relationships.values() for i in v)
    ):
        raise ValueError(
            "tf_relationships must be a dictionary between ints and lists of ints"
        )
    if not all(
        k in range(binding_enrichment_data.shape[1]) for k in tf_relationships.keys()
    ) or not all(
        i in range(binding_enrichment_data.shape[1])
        for v in tf_relationships.values()
        for i in v
    ):
        raise ValueError(
            "all keys and values in tf_relationships must be within the \
                  bounds of the binding_data tensor's number of TFs"
        )
    if not len(tf_relationships) == binding_enrichment_data.shape[1]:
        raise ValueError(
            "tf_relationships must have the same number of TFs as the \
                binding_data tensor passed into the function"
        )

    # Extract bound/unbound labels and enrichment scores
    bound_labels = binding_enrichment_data[:, :, 0]  # shape: (num_genes, num_tfs)
    enrichment_scores = binding_enrichment_data[:, :, 1]  # shape: (num_genes, num_tfs)

    # we set all unbound scores to 0, then we will go through and also
    # set any bound scores to unbound_mean if the related tfs are not also bound
    adjusted_mean_matrix = torch.where(
        bound_labels == 1, enrichment_scores, torch.zeros_like(enrichment_scores)
    )  # shape: (num_genes, num_tfs)

    for gene_idx in range(bound_labels.shape[0]):
        for tf_index, related_tfs in tf_relationships.items():
            if bound_labels[gene_idx, tf_index] == 1 and torch.all(
                bound_labels[gene_idx, related_tfs] == 1
            ):
                # OLD: adjustment_multiplier = torch.rand(1)
                # divide its enrichment score by the maximum magnitude possible to
                # create an adjustment multipler that scales with increasing enrichment
                adjustment_multiplier = enrichment_scores[gene_idx, tf_index] / abs(
                    enrichment_scores.max()
                )

                # randomly adjust the gene by some portion of the max adjustment
                adjusted_mean_matrix[gene_idx, tf_index] = bound_mean + (
                    adjustment_multiplier * max_adjustment
                )
            else:
                # related tfs are not all bound, set the enrichment score to unbound
                # mean
                adjusted_mean_matrix[gene_idx, tf_index] = unbound_mean

    return adjusted_mean_matrix  # shape (num_genes, num_tfs)


def generate_perturbation_effects(
    binding_data: torch.Tensor,
    tf_index: int | None = None,
    unbound_mean: float = 0.0,
    unbound_std: float = 1.0,
    bound_mean: float = 3.0,
    bound_std: float = 1.0,
    max_mean_adjustment: float = 0.0,
    adjustment_function: Callable[
        [torch.Tensor, float, float, float], torch.Tensor
    ] = default_perturbation_effect_adjustment_function,
    **kwargs,
) -> torch.Tensor:
    """
    Generate perturbation effects for genes.

    If `max_mean_adjustment` is greater than 0, then the mean of the
    effects are adjusted based on the binding_data and the function passed
    in `adjustment_function`. See `default_perturbation_effect_adjustment_function()`
    for the default option. If `max_mean_adjustment` is 0, then the mean
    is not adjusted. Additional keyword arguments may be passed in that will be
    passed along to the adjustment function.

    :param binding_data: A tensor of binding data with dimensions [n_genes, n_tfs, 3]
        where the entries in the third dimension are a matrix with columns
        [label, enrichment, pvalue].
    :type binding_data: torch.Tensor
    :param tf_index: The index of the TF in the binding_data tensor. Not used if we
        are adjusting the means (ie only used if max_mean_adjustment == 0).
        Defaults to None
    :type tf_index: int
    :param unbound_mean: The mean for unbound genes. Defaults to 0.0
    :type unbound_mean: float, optional
    :param unbound_std: The standard deviation for unbound genes. Defaults to 1.0
    :type unbound_std: float, optional
    :param bound_mean: The mean for bound genes. Defaults to 3.0
    :type bound_mean: float, optional
    :param bound_std: The standard deviation for bound genes. Defaults to 1.0
    :type bound_std: float, optional
    :param max_mean_adjustment: The maximum adjustment to the base mean based
        on enrichment. Defaults to 0.0
    :type max_mean_adjustment: float, optional

    :return: A tensor of perturbation effects for each gene.
    :rtype: torch.Tensor

    :raises ValueError: If binding_data is not a 3D tensor with the third
        dimension having a length of 3
    :raises ValueError: If unbound_mean, unbound_std, bound_mean, bound_std,
        or max_mean_adjustment are not floats

    """
    # check that a valid combination of inputs has been passed in
    if max_mean_adjustment == 0.0 and tf_index is None:
        raise ValueError("If max_mean_adjustment is 0, then tf_index must be specified")

    if binding_data.ndim != 3 or binding_data.shape[2] != 3:
        raise ValueError(
            "enrichment_tensor must have dimensions [num_genes, num_TFs, "
            "[label, enrichment, pvalue]]"
        )
    # check the rest of the inputs
    if not all(
        isinstance(i, float)
        for i in (unbound_mean, unbound_std, bound_mean, bound_std, max_mean_adjustment)
    ):
        raise ValueError(
            "unbound_mean, unbound_std, bound_mean, bound_std, "
            "and max_mean_adjustment must be floats"
        )
    # check the Callable signature
    if not all(
        i in inspect.signature(adjustment_function).parameters
        for i in (
            "binding_enrichment_data",
            "bound_mean",
            "unbound_mean",
            "max_adjustment",
        )
    ):
        raise ValueError(
            "adjustment_function must have the signature "
            "(binding_enrichment_data, bound_mean, unbound_mean, max_adjustment)"
        )

    # Initialize an effects tensor for all genes
    effects = torch.empty(
        binding_data.size(0), dtype=torch.float32, device=binding_data.device
    )

    # Randomly assign signs for each gene
    # fmt: off
    signs = torch.randint(0, 2, (effects.size(0),),
                          dtype=torch.float32,
                          device=binding_data.device) * 2 - 1
    # fmt: on

    # Apply adjustments to the base mean for the bound genes, if necessary
    if max_mean_adjustment > 0 and adjustment_function is not None:
        # Assuming adjustment_function returns a vector of means for each gene.
        # bound genes that meet the criteria for adjustment will be affected by
        # the status of the TFs. What TFs affect a given gene must be specified by
        # the adjustment_function()
        adjusted_means = adjustment_function(
            binding_data,
            bound_mean,
            unbound_mean,
            max_mean_adjustment,
            **kwargs,
        )

        # add adjustments, ensuring they respect the original sign
        if adjusted_means.ndim == 1:
            effects = signs * torch.abs(
                torch.normal(mean=adjusted_means, std=bound_std)
            )
        else:
            effects = torch.zeros_like(adjusted_means)
            for col_idx in range(effects.size(1)):
                effects[:, col_idx] = signs * torch.abs(
                    torch.normal(mean=adjusted_means[:, col_idx], std=bound_std)
                )
    else:
        bound_mask = binding_data[:, tf_index, 0] == 1

        # Generate effects based on the unbound and bound means, applying the sign
        effects[~bound_mask] = signs[~bound_mask] * torch.abs(
            torch.normal(
                mean=unbound_mean, std=unbound_std, size=(torch.sum(~bound_mask),)
            )
        )
        effects[bound_mask] = signs[bound_mask] * torch.abs(
            torch.normal(mean=bound_mean, std=bound_std, size=(torch.sum(bound_mask),))
        )

    return effects
