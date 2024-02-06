import logging

import pandas as pd
import torch

logger = logging.getLogger(__name__)


def generate_gene_population(
    total: int = 1000, signal_group: float = 0.3
) -> torch.Tensor:
    """
    Generate two sets of genes, one of which will be considered genes which show a
    signal to both TF binding and response, and the other which does not. The return is
    a tensor where the first column is the gene/feature identifier (0 to total-1) and
    the second column is binary indicating whether the gene is in the signal group or
    not.

    :param total: The total number of genes. defaults to 1000
    :type total: int, optional
    :param signal_group: The proportion of genes in the signal group. defaults to 0.3
    :type signal_group: float, optional
    :return: A tensor where the first column is the gene/feature identifier and the
        second column is binary indicating whether the gene is in the signal group or
        not.
    :rtype: torch.Tensor
    :raises ValueError: if total is not an integer
    :raises ValueError: If signal_group is not between 0 and 1

    """
    if not isinstance(total, int):
        raise ValueError("total must be an integer")
    if not 0 <= signal_group <= 1:
        raise ValueError("signal_group must be between 0 and 1")

    signal_group_size = int(total * signal_group)
    logger.info("Generating %s genes with signal", signal_group_size)

    # Generating gene identifiers
    gene_ids = torch.arange(total, dtype=torch.int32)

    # Generating binary labels for signal group
    labels = torch.cat(
        (
            torch.ones(signal_group_size, dtype=torch.int32),
            torch.zeros(total - signal_group_size, dtype=torch.int32),
        )
    )

    # Randomly shuffling labels
    shuffled_indices = torch.randperm(total)
    shuffled_labels = labels[shuffled_indices]

    # Combining gene IDs and their labels
    gene_population = torch.stack((gene_ids, shuffled_labels), dim=1)

    return gene_population


def generate_perturbation_effects(
    total: int,
    signal_group_size: int,
    unaffected_mean: float,
    unaffected_std: float,
    affected_mean: float,
    affected_std: float,
) -> torch.Tensor:
    """
    Generate perturbation effects for genes.

    See generate_perturbation_binding_data() for more details.

    :raises ValueError: If signal_group_size is not less than total

    """
    if signal_group_size > total:
        raise ValueError("Signal group size must not exceed total")

    unaffected_group_size = total - signal_group_size

    unaffected_perturbation_effect = torch.cat(
        (
            torch.normal(
                unaffected_mean, unaffected_std, size=(unaffected_group_size // 2,)
            ),
            torch.normal(
                -unaffected_mean, unaffected_std, size=(unaffected_group_size // 2,)
            ),
        )
    )

    affected_perturbation_effect = torch.cat(
        (
            torch.normal(affected_mean, affected_std, size=(signal_group_size // 2,)),
            torch.normal(-affected_mean, affected_std, size=(signal_group_size // 2,)),
        )
    )

    perturbation_effect = torch.cat(
        (unaffected_perturbation_effect, affected_perturbation_effect)
    )
    return perturbation_effect


def generate_binding_effects(
    total: int, signal_group_size: int, unaffected_lambda: float, affected_lambda: float
) -> torch.Tensor:
    """
    Generate binding effects for genes.

    see generate_perturbation_binding_data() for more details.

    :raises ValueError: If unaffected_lambda or affected_lambda is not non-negative
    :raises ValueError: If signal_group_size is not less than total

    """
    if unaffected_lambda < 0 or affected_lambda < 0:
        raise ValueError("Lambda values must be non-negative")
    if signal_group_size > total or signal_group_size < 0:
        raise ValueError("Signal group size must be less than total")

    unaffected_group_size = total - signal_group_size

    unaffected_binding_effect = torch.poisson(
        torch.full((unaffected_group_size,), unaffected_lambda)
    )
    affected_binding_effect = torch.poisson(
        torch.full((signal_group_size,), affected_lambda)
    )

    binding_effect = torch.cat((unaffected_binding_effect, affected_binding_effect))
    return binding_effect


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


def generate_perturbation_binding_data(
    gene_population: torch.Tensor,
    unaffected_perturbation_abs_mean: float = 0.0,
    unaffected_perturbation_std: float = 1.0,
    affected_perturbation_abs_mean: float = 3.0,
    affected_perturbation_std: float = 1.0,
    unaffected_binding_lambda: float = 1e-3,
    affected_binding_lambda: float = 3.0,
) -> pd.DataFrame:
    """
    Using a normal distribution for the perturbation effect, a poisson distribution for
    the binding effect, simulate the perturbation and binding data. Note that for the
    perturbation data, the affected and unaffected genes are divided into half where one
    half has a positive perturbation_mean and the other has a negative perturbation_mean
    in order to simulate both up and down regulation. Pvalues are calculated from a
    random distribution based on their effect size, with the assumption that larger
    effects are less likely to be false positives.

    :param gene_population: A tensor where the first column is the gene/feature
        identifier and the second column is binary indicating whether the gene
        is in the signal group or not. See generate_gene_population() for
        more details.
    :type gene_population: torch.Tensor
    :param unaffected_perturbation_abs_mean: The absolute mean of the
        perturbation effect for the unaffected genes. defaults to 0.0
    :type unaffected_perturbation_abs_mean: float, optional
    :param unaffected_perturbation_std: The standard deviation of the
        perturbation effect for the unaffected genes. defaults to 1.0
    :type unaffected_perturbation_std: float, optional
    :param affected_perturbation_abs_mean: The absolute mean of the
        perturbation effect for the affected genes. defaults to 3.0
    :type affected_perturbation_abs_mean: float, optional
    :param affected_perturbation_std: The standard deviation of the
        perturbation effect for the affected genes. defaults to 1.0
    :type affected_perturbation_std: float, optional
    :param unaffected_binding_lambda: The lambda parameter for the poisson
        distribution for the unaffected genes. defaults to 1e-3
    :type unaffected_binding_lambda: float, optional
    :param affected_binding_lambda: The lambda parameter for the poisson
        distribution for the affected genes. defaults to 3.0
    :type affected_binding_lambda: float, optional

    :return: A dataframe containing the following columns:
        gene_id: (str) The gene identifier
        signal: (boolean) Whether the gene is in the signal group or not
        expression_effect: (float) The perturbation effect
        expression_pvalue: (float) The pvalue of the perturbation effect
        binding_effect: (float) The binding effect
        binding_pvalue: (float) The pvalue of the binding effect
    :rtype: pd.DataFrame

    :raises ValueError: If gene_population is not a tensor with two columns
        where the second column is binary
    :raises ValueError: If unaffected_perturbation_abs_mean is not a float
    :raises ValueError: If unaffected_perturbation_std is not a float
    :raises ValueError: If affected_perturbation_abs_mean is not a float
    :raises ValueError: If affected_perturbation_std is not a float
    :raises ValueError: If unaffected_binding_lambda is not a float or <= 0
    :raises ValueError: If affected_binding_lambda is not a float or <= 0

    """
    # check inputs
    if not isinstance(gene_population, torch.Tensor):
        raise ValueError("gene_population must be a tensor")
    if gene_population.shape[1] != 2:
        raise ValueError("gene_population must have two columns")
    if gene_population.dtype != torch.int32 and gene_population.dtype != torch.int64:
        raise ValueError("gene_population must have torch.int32 or torch.int64 dtype")
    if gene_population.shape[0] == 0:
        raise ValueError("gene_population must have at least one row")
    if not torch.all((gene_population[:, 1] == 0) | (gene_population[:, 1] == 1)):
        raise ValueError("gene_population second column must be binary")
    if not isinstance(unaffected_perturbation_abs_mean, float):
        raise ValueError("unaffected_perturbation_abs_mean must be a float")
    if not isinstance(unaffected_perturbation_std, float):
        raise ValueError("unaffected_perturbation_std must be a float")
    if not isinstance(affected_perturbation_abs_mean, float):
        raise ValueError("affected_perturbation_abs_mean must be a float")
    if not isinstance(affected_perturbation_std, float):
        raise ValueError("affected_perturbation_std must be a float")
    if not isinstance(unaffected_binding_lambda, float):
        raise ValueError("unaffected_binding_lambda must be a float")
    if unaffected_binding_lambda <= 0:
        raise ValueError("unaffected_binding_lambda must be > 0")
    if not isinstance(affected_binding_lambda, float):
        raise ValueError("affected_binding_lambda must be a float")
    if affected_binding_lambda <= 0:
        raise ValueError("affected_binding_lambda must be > 0")

    total = gene_population.shape[0]
    signal_group_size = torch.sum(gene_population[:, 1]).item()

    # Generate effects
    perturbation_effect = generate_perturbation_effects(
        total,
        signal_group_size,
        unaffected_perturbation_abs_mean,
        unaffected_perturbation_std,
        affected_perturbation_abs_mean,
        affected_perturbation_std,
    )
    binding_effect = generate_binding_effects(
        total, signal_group_size, unaffected_binding_lambda, affected_binding_lambda
    )

    # Generate p-values
    perturbation_pvalues = generate_pvalues(perturbation_effect)
    binding_pvalues = generate_pvalues(binding_effect)

    # Combine into DataFrame and return
    df = pd.DataFrame(
        {
            "gene_id": gene_population[:, 0].numpy(),
            "signal": gene_population[:, 1].numpy().astype(bool),
            "expression_effect": perturbation_effect.numpy(),
            "expression_pvalue": perturbation_pvalues.numpy(),
            "binding_effect": binding_effect.numpy(),
            "binding_pvalue": binding_pvalues.numpy(),
        }
    )

    return df
