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


def default_perturbation_effect_adjustment_function_old(
    binding_enrichment_data: torch.Tensor,
    signal_mean: float,
    noise_mean: float,
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
    :param signal_mean: The mean for signal genes.
    :type signal_mean: float
    :param noise_mean: The mean for noise genes.
    :type noise_mean: float
    :param max_adjustment: The maximum adjustment to the base mean based on enrichment.
    :type max_adjustment: float
    :param tf_relationships: Unused in this function. It is only here to match the
        signature of the other adjustment functions.
    :type tf_relationships: dict[int, list[int]], optional
    :return: Adjusted mean as a tensor.
    :rtype: torch.Tensor

    """
    # Extract signal/noise labels and enrichment scores
    signal_labels = binding_enrichment_data[:, :, 0]
    enrichment_scores = binding_enrichment_data[:, :, 1]

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

    # NOTE: setting the noise genes to the noise mean
    adjusted_mean[signal_labels[:, 0] == 0] = noise_mean

    return adjusted_mean


def default_perturbation_effect_adjustment_function(
    binding_enrichment_data: torch.Tensor,
    signal_mean: float,
    noise_mean: float,
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
    :param signal_mean: The mean for signal genes.
    :type signal_mean: float
    :param noise_mean: The mean for noise genes.
    :type noise_mean: float
    :param max_adjustment: The maximum adjustment to the base mean based on enrichment.
    :type max_adjustment: float
    :param tf_relationships: Unused in this function. It is only here to match the
        signature of the other adjustment functions.
    :type tf_relationships: dict[int, list[int]], optional
    :return: Adjusted mean as a tensor.
    :rtype: torch.Tensor

    """
    # Extract signal/noise labels and enrichment scores
    signal_labels = binding_enrichment_data[:, :, 0]
    enrichment_scores = binding_enrichment_data[:, :, 1]

    adjusted_mean_matrix = torch.where(
        signal_labels == 1, enrichment_scores, torch.zeros_like(enrichment_scores)
    )

    for gene_idx in range(signal_labels.shape[0]):
        for tf_index in range(signal_labels.shape[1]):
            if signal_labels[gene_idx, tf_index] == 1:
                # draw a random value between 0 and 1 to use to control
                # magnitude of adjustment
                adjustment_multiplier = torch.rand(1)

                # randomly adjust the gene by some portion of the max adjustment
                adjusted_mean_matrix[gene_idx, tf_index] = signal_mean + (
                    adjustment_multiplier * max_adjustment
                )
            else:
                # related tfs are not all bound, so set the enrichment
                # score to noise mean
                adjusted_mean_matrix[gene_idx, tf_index] = noise_mean

    return adjusted_mean_matrix


def perturbation_effect_adjustment_function_with_tf_relationships_boolean_logic(
    binding_enrichment_data: torch.Tensor,
    signal_mean: float,
    noise_mean: float,
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
    :param signal_mean: The mean for signal genes.
    :type signal_mean: float
    :param noise_mean: The mean for noise genes.
    :type noise_mean: float
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

    # Extract signal/noise labels and enrichment scores
    signal_labels = binding_enrichment_data[:, :, 0]  # shape: (num_genes, num_tfs)
    enrichment_scores = binding_enrichment_data[:, :, 1]  # shape: (num_genes, num_tfs)

    # we set all unbound scores to 0, then we will go through and also set any
    # bound scores to noise_mean if the related boolean statements are not satisfied
    adjusted_mean_matrix = torch.where(
        signal_labels == 1, enrichment_scores, torch.zeros_like(enrichment_scores)
    )  # shape: (num_genes, num_tfs)

    for gene_idx in range(signal_labels.shape[0]):
        for tf_index, relations in tf_relationships.items():
            # check if all relations (boolean relationships)
            # associated with TFs are satisfied
            if signal_labels[gene_idx, tf_index] == 1 and all(
                relation.evaluate(signal_labels[gene_idx].tolist())
                for relation in relations
            ):
                # draw a random value between 0 and 1 to use to
                # control magnitude of adjustment
                adjustment_multiplier = torch.rand(1)

                # randomly adjust the gene by some portion of the max adjustment
                adjusted_mean_matrix[gene_idx, tf_index] = signal_mean + (
                    adjustment_multiplier * max_adjustment
                )
            else:
                # related tfs are not all bound, set the enrichment score to noise mean
                adjusted_mean_matrix[gene_idx, tf_index] = noise_mean

    return adjusted_mean_matrix  # shape (num_genes, num_tfs)


def perturbation_effect_adjustment_function_with_tf_relationships(
    binding_enrichment_data: torch.Tensor,
    signal_mean: float,
    noise_mean: float,
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
    :param signal_mean: The mean for signal genes.
    :type signal_mean: float
    :param noise_mean: The mean for noise genes.
    :type noise_mean: float
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

    # Extract signal/noise labels and enrichment scores
    signal_labels = binding_enrichment_data[:, :, 0]  # shape: (num_genes, num_tfs)
    enrichment_scores = binding_enrichment_data[:, :, 1]  # shape: (num_genes, num_tfs)

    # we set all unbound scores to 0, then we will go through and also
    # set any bound scores to noise_mean if the related tfs are not also bound
    adjusted_mean_matrix = torch.where(
        signal_labels == 1, enrichment_scores, torch.zeros_like(enrichment_scores)
    )  # shape: (num_genes, num_tfs)

    for gene_idx in range(signal_labels.shape[0]):
        for tf_index, related_tfs in tf_relationships.items():
            if signal_labels[gene_idx, tf_index] == 1 and torch.all(
                signal_labels[gene_idx, related_tfs] == 1
            ):
                # draw a random value between 0 and 1 to use to
                # control magnitude of adjustment
                adjustment_multiplier = torch.rand(1)

                # randomly adjust the gene by some portion of the max adjustment
                adjusted_mean_matrix[gene_idx, tf_index] = signal_mean + (
                    adjustment_multiplier * max_adjustment
                )
            else:
                # related tfs are not all bound, set the enrichment score to noise mean
                adjusted_mean_matrix[gene_idx, tf_index] = noise_mean

    return adjusted_mean_matrix  # shape (num_genes, num_tfs)


def generate_perturbation_effects(
    binding_data: torch.Tensor,
    tf_index: int,
    noise_mean: float = 0.0,
    noise_std: float = 1.0,
    signal_mean: float = 3.0,
    signal_std: float = 1.0,
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
    :param tf_index: The index of the TF in the binding_data tensor.
    :type tf_index: int
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
    # check the Callable signature
    if not all(
        i in inspect.signature(adjustment_function).parameters
        for i in (
            "binding_enrichment_data",
            "signal_mean",
            "noise_mean",
            "max_adjustment",
        )
    ):
        raise ValueError(
            "adjustment_function must have the signature "
            "(binding_enrichment_data, signal_mean, noise_mean, max_adjustment)"
        )

    signal_mask = binding_data[:, tf_index, 0] == 1

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
        # Assuming adjustment_function returns a vector of means for each gene.
        # Signal genes that meet the criteria for adjustment will be affected by
        # the status of the TFs. What TFs affect a given gene must be specified by
        # the adjustment_function()
        adjusted_means = adjustment_function(
            binding_data,
            signal_mean,
            noise_mean,
            max_mean_adjustment,
            **kwargs,
        )

        # add adjustments, ensuring they respect the original sign
        if adjusted_means.ndim == 1:
            effects = signs * torch.abs(
                torch.normal(mean=adjusted_means, std=signal_std)
            )
        else:
            effects = torch.zeros_like(adjusted_means)
            for col_idx in range(effects.size(1)):
                effects[:, col_idx] = signs * torch.abs(
                    torch.normal(mean=adjusted_means[:, col_idx], std=signal_std)
                )
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
