import logging
from collections.abc import Callable

import torch

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
    total: int = 1000, signal_group: float = 0.3
) -> GenePopulation:
    """
    Generate two sets of genes, one of which will be considered genes which show a
    signal, and the other which does not. The return is a one dimensional boolean tensor
    where a value of '0' means that the gene at that index is part of the noise group
    and a '1' means the gene at that index is part of the signal group. The length of
    the tensor is the number of genes in this simulated organism.

    :param total: The total number of genes. defaults to 1000
    :type total: int, optional
    :param signal_group: The proportion of genes in the signal group. defaults to 0.3
    :type signal_group: float, optional
    :return: A one dimensional tensor of boolean values where the set of indices with a
        value of '1' are the signal group and the set of indices with a value of '0' are
        the noise group.
    :rtype: GenePopulation
    :raises TypeError: if total is not an integer
    :raises ValueError: If signal_group is not between 0 and 1

    """
    if not isinstance(total, int):
        raise TypeError("total must be an integer")
    if not 0 <= signal_group <= 1:
        raise ValueError("signal_group must be between 0 and 1")

    signal_group_size = int(total * signal_group)
    logger.info("Generating %s genes with signal", signal_group_size)

    labels = torch.cat(
        (
            torch.ones(signal_group_size, dtype=torch.bool),
            torch.zeros(total - signal_group_size, dtype=torch.bool),
        )
    )[torch.randperm(total)]

    return GenePopulation(labels)


def generate_binding_effects(
    gene_population: GenePopulation,
    background_hops_range: tuple[int, int] = (1, 100),
    noise_experiment_hops_range: tuple[int, int] = (0, 1),
    signal_experiment_hops_range: tuple[int, int] = (1, 6),
    total_background_hops: int = 1000,
    total_experiment_hops: int = 76,
    pseudocount: float = 1e-10,
) -> torch.Tensor:
    """
    Generate enrichment effects for genes using vectorized operations, based on their
    signal designation, with separate experiment hops ranges for noise and signal genes.

    Note that the default values are a scaled down version of actual data. See also
    https://github.com/cmatKhan/callingCardsTools/blob/main/callingcardstools/PeakCalling/yeast/enrichment.py

    :param gene_population: A GenePopulation object. See `generate_gene_population()`
    :type gene_population: GenePopulation
    :param background_hops_range: The range of hops for background genes. Defaults to
        (1, 100)
    :type background_hops_range: Tuple[int, int], optional
    :param noise_experiment_hops_range: The range of hops for noise genes. Defaults to
        (0, 1)
    :type noise_experiment_hops_range: Tuple[int, int], optional
    :param signal_experiment_hops_range: The range of hops for signal genes. Defaults to
        (1, 6)
    :type signal_experiment_hops_range: Tuple[int, int], optional
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
    :raises TypeError: If noise_experiment_hops_range is not a tuple
    :raises TypeError: If signal_experiment_hops_range is not a tuple
    :raises ValueError: If background_hops_range is not a tuple of length 2
    :raises ValueError: If noise_experiment_hops_range is not a tuple of length 2
    :raises ValueError: If signal_experiment_hops_range is not a tuple of length 2

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
        "noise_experiment_hops_range": noise_experiment_hops_range,
        "signal_experiment_hops_range": signal_experiment_hops_range,
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

    # Generate experiment hops noise genes
    noise_experiment_hops = torch.randint(
        low=noise_experiment_hops_range[0],
        high=noise_experiment_hops_range[1] + 1,
        size=(gene_population.labels.shape[0],),
    )
    # Generate experiment hops signal genes
    signal_experiment_hops = torch.randint(
        low=signal_experiment_hops_range[0],
        high=signal_experiment_hops_range[1] + 1,
        size=(gene_population.labels.shape[0],),
    )

    # Use signal designation to select appropriate experiment hops
    experiment_hops = torch.where(
        gene_population.labels == 1, signal_experiment_hops, noise_experiment_hops
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
    signal_mean: float,
    noise_mean: float,
    max_adjustment: float,
) -> torch.Tensor:
    """
    Default function to adjust the mean of the perturbation effect based on the
    enrichment score.

    All functions that are passed to generate_perturbation_data() in the argument
    adjustment_function must have the same signature as this function.

    :param binding_enrichment_data: A tensor of enrichment scores for each gene with
        dimensions [n_genes, n_tfs, 3] where the entries in the third dimension are a
        matrix with columns [label, enrichment, pvalue].
    :type binding_enrichment_data: torch.Tensor
    :param signal_mean: The mean for signal genes.
    :type signal_mean: float
    :param noise_mean: The mean for noise genes.
    :type noise_mean: float
    :param max_adjustment: The maximum adjustment to the base mean based on enrichment.
    :return: Adjusted mean as a tensor.
    :rtype: torch.Tensor

    """
    # Extract signal/noise labels and enrichment scores
    signal_labels = binding_enrichment_data[:, :, 0]
    enrichment_scores = binding_enrichment_data[:, :, 1]

    # Set noise (label 0) enrichment scores to 0 and then sum across TFs
    summed_enrichment_scores = torch.where(
        signal_labels == 1, enrichment_scores, torch.zeros_like(enrichment_scores)
    ).sum(dim=1)

    # Normalize and transform summed enrichment scores
    scaled_scores = (summed_enrichment_scores - summed_enrichment_scores.min()) / (
        summed_enrichment_scores.max() - summed_enrichment_scores.min()
    )

    # Apply a moderate exponential transformation to increase small differences
    transformed_scores = torch.sqrt(scaled_scores)

    adjusted_scores = transformed_scores * max_adjustment

    # Generate adjustment factors for signal genes, ensuring a value of 0 for
    # noise genes Assuming the signal/noise label is consistent across TFs for
    # each gene
    adjusted_mean = signal_mean + torch.where(
        signal_labels[:, 0] == 1, adjusted_scores, torch.zeros_like(adjusted_scores)
    )

    adjusted_mean[signal_labels[:, 0] == 0] = noise_mean

    return adjusted_mean


def generate_perturbation_effects(
    binding_data: torch.Tensor,
    noise_mean: float = 0.0,
    noise_std: float = 1.0,
    signal_mean: float = 3.0,
    signal_std: float = 1.0,
    max_mean_adjustment: float = 0.0,
    adjustment_function: Callable[
        [torch.Tensor, float, float, float], torch.Tensor
    ] = default_perturbation_effect_adjustment_function,
) -> torch.Tensor:
    """
    Generate perturbation effects for genes.

    If `max_mean_adjustment` is greater than 0, then the mean of the
    effects are adjusted based on the binding_data and the function passed
    in `adjustment_function`. See `default_perturbation_effect_adjustment_function()`
    for the default option. If `max_mean_adjustment` is 0, then the mean
    is not adjusted.

    :param binding_data: A tensor of binding data with dimensions [n_genes, n_tfs, 3]
        where the entries in the third dimension are a matrix with columns
        [label, enrichment, pvalue].
    :type binding_data: torch.Tensor
    :param noise_mean: The mean for noise genes. Defaults to 0.0
    :type noise_mean: float, optional
    :param noise_std: The standard deviation for noise genes. Defaults to 1.0
    :type noise_std: float, optional
    :param signal_mean: The mean for signal genes. Defaults to 3.0
    :type signal_mean: float, optional
    :param signal_std: The standard deviation for signal genes. Defaults to 1.0
    :type signal_std: float, optional
    :param max_mean_adjustment: The maximum adjustment to the base mean based
        on enrichment. Defaults to 0.0
    :type max_mean_adjustment: float, optional

    :return: A tensor of perturbation effects for each gene.
    :rtype: torch.Tensor

    :raises ValueError: If binding_data is not a 3D tensor with the third
        dimension having a length of 3
    :raises ValueError: If noise_mean, noise_std, signal_mean, signal_std,
        or max_mean_adjustment are not floats

    """
    if binding_data.ndim != 3 or binding_data.shape[2] != 3:
        raise ValueError(
            "enrichment_tensor must have dimensions [num_genes, num_TFs, "
            "[label, enrichment, pvalue]]"
        )
    # check the rest of the inputs
    if not all(
        isinstance(i, float)
        for i in (noise_mean, noise_std, signal_mean, signal_std, max_mean_adjustment)
    ):
        raise ValueError(
            "noise_mean, noise_std, signal_mean, signal_std, "
            "and max_mean_adjustment must be floats"
        )

    signal_mask = (binding_data[:, :, 0] == 1).any(dim=1)

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

    # Apply adjustments to the base mean for the signal genes, if necessary
    if max_mean_adjustment > 0 and adjustment_function is not None:
        # Assuming adjustment_function returns an adjustment factor for each gene
        adjusted_means = adjustment_function(
            binding_data, signal_mean, noise_mean, max_mean_adjustment
        )
        # add adjustments, ensuring they respect the original sign
        effects = signs * torch.abs(torch.normal(mean=adjusted_means, std=signal_std))
    else:
        # Generate effects based on the noise and signal means, applying the sign
        effects[~signal_mask] = signs[~signal_mask] * torch.abs(
            torch.normal(
                mean=noise_mean, std=noise_std, size=(torch.sum(~signal_mask),)
            )
        )
        effects[signal_mask] = signs[signal_mask] * torch.abs(
            torch.normal(
                mean=signal_mean, std=signal_std, size=(torch.sum(signal_mask),)
            )
        )

    return effects
