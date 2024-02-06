# mypy: disable-error-code=arg-type
import pandas as pd
import pytest
import torch

from yeastdnnexplorer.probability_models.generate_data import (
    generate_binding_effects,
    generate_gene_population,
    generate_perturbation_binding_data,
    generate_perturbation_effects,
    generate_pvalues,
)


def test_generate_gene_population():
    total_genes = 1000
    signal_ratio = 0.3
    signal_group_size = int(total_genes * signal_ratio)

    gene_population = generate_gene_population(total_genes, signal_ratio)

    # Check if the output is a 2D tensor
    assert gene_population.ndim == 2

    # Check if the output has the correct shape
    assert gene_population.shape == (total_genes, 2)

    # Check if the first column contains identifiers 0 to total-1
    assert all(gene_population[:, 0] == torch.arange(total_genes))

    # Check if the second column contains the correct number of signal
    # and non-signal genes
    assert torch.sum(gene_population[:, 1]) == signal_group_size
    assert torch.sum(gene_population[:, 1] == 0) == total_genes - signal_group_size

    # Additional tests could include checking the datatype of the tensor elements
    assert gene_population.dtype == torch.int32


@pytest.mark.parametrize("total, ratio", [(1000, 0.3), (500, 0.5), (2000, 0.1)])
def test_gene_population(total, ratio):
    gene_population = generate_gene_population(total, ratio)
    signal_group_size = int(total * ratio)

    assert gene_population.shape == (total, 2)
    assert torch.sum(gene_population[:, 1]) == signal_group_size
    assert torch.sum(gene_population[:, 1] == 0) == total - signal_group_size


def test_gene_population_invalid_input():
    with pytest.raises(ValueError):
        # invalid string input
        generate_gene_population(total="1000", signal_group=0.3)

    with pytest.raises(ValueError):
        generate_gene_population(total=1000, signal_group=1.2)

    with pytest.raises(ValueError):
        generate_perturbation_binding_data(torch.rand((100, 1)))  # Invalid shape

    with pytest.raises(ValueError):
        generate_perturbation_binding_data(torch.rand((0, 2)))  # Empty tensor

    with pytest.raises(ValueError):
        generate_perturbation_binding_data(torch.rand((100, 2)))  # Non-integer tensor

    with pytest.raises(ValueError):
        generate_perturbation_binding_data(
            torch.tensor([[1, -1], [2, 2]], dtype=torch.int32)
        )


def test_generate_perturbation_effects_valid_inputs():
    total = 100
    signal_group_size = 50
    unaffected_mean = 1.0
    unaffected_std = 0.5
    affected_mean = 2.0
    affected_std = 0.7

    effects = generate_perturbation_effects(
        total,
        signal_group_size,
        unaffected_mean,
        unaffected_std,
        affected_mean,
        affected_std,
    )

    # Check if the returned tensor has the correct shape
    assert effects.shape[0] == total, (
        "The number of effects generated " "does not match the total"
    )

    # Check if the returned object is a tensor
    assert isinstance(effects, torch.Tensor), "Returned object is not a tensor"


def test_generate_perturbation_effects_invalid_inputs():
    # Test with negative total
    with pytest.raises(ValueError):
        generate_perturbation_effects(-100, 50, 1.0, 0.5, 2.0, 0.7)

    # Test with signal group size greater than total
    with pytest.raises(ValueError):
        generate_perturbation_effects(50, 100, 1.0, 0.5, 2.0, 0.7)

    # Test with non-numeric mean or standard deviation
    with pytest.raises(TypeError):
        generate_perturbation_effects(
            100, 50, "invalid", 0.5, 2.0, 0.7
        )  # mypy: ignore arg-type # noqa

    with pytest.raises(TypeError):
        generate_perturbation_effects(100, 50, 1.0, "invalid", 2.0, 0.7)

    with pytest.raises(TypeError):
        generate_perturbation_effects(100, 50, 1.0, 0.5, "invalid", 0.7)

    with pytest.raises(TypeError):
        generate_perturbation_effects(100, 50, 1.0, 0.5, 2.0, "invalid")


def test_generate_pvalues_valid_input():
    # Setup test data
    effects = torch.randn(100)  # Random tensor of size 100
    large_effect_percentile = 0.9
    large_effect_upper_pval = 0.2

    # Call the function
    pvalues = generate_pvalues(
        effects, large_effect_percentile, large_effect_upper_pval
    )

    # Check if the output is a tensor
    assert isinstance(pvalues, torch.Tensor)

    # Check if all p-values are within the range [0, 1]
    assert torch.all(pvalues >= 0) and torch.all(pvalues <= 1)

    # Check if larger effects have p-values within the specified upper bound
    large_effect_threshold = torch.quantile(torch.abs(effects), large_effect_percentile)
    large_effect_mask = torch.abs(effects) >= large_effect_threshold
    assert torch.all(pvalues[large_effect_mask] <= large_effect_upper_pval)


def test_generate_pvalues_invalid_input():
    # Test with non-tensor input
    with pytest.raises(ValueError):
        generate_pvalues([0.5, 0.3, 0.7])  # Invalid input as list

    # Test with non-numeric tensor
    with pytest.raises(ValueError):
        generate_pvalues(
            torch.tensor(["a", "b", "c"])
        )  # Invalid input as non-numeric tensor


def test_generate_binding_effects_valid_input():
    total = 100
    signal_group_size = 30
    unaffected_lambda = 2.0
    affected_lambda = 5.0

    # Call the function
    binding_effect = generate_binding_effects(
        total, signal_group_size, unaffected_lambda, affected_lambda
    )

    # Check if the output is a tensor
    assert isinstance(binding_effect, torch.Tensor)

    # Check if the output size is correct
    assert binding_effect.shape[0] == total

    # Check if the first part corresponds to unaffected group
    assert torch.all(binding_effect[: total - signal_group_size] >= 0)

    # Check if the second part corresponds to affected group
    assert torch.all(binding_effect[total - signal_group_size :] >= 0)


@pytest.mark.parametrize(
    "total, signal_group_size, unaffected_lambda, affected_lambda",
    [
        (-10, 5, 2.0, 5.0),  # Negative total
        (100, -5, 2.0, 5.0),  # Negative signal group size
        (100, 150, 2.0, 5.0),  # Signal group size larger than total
        (100, 50, -2.0, 5.0),  # Negative unaffected lambda
        (100, 50, 2.0, -5.0),  # Negative affected lambda
    ],
)
def test_generate_binding_effects_invalid_input(
    total, signal_group_size, unaffected_lambda, affected_lambda
):
    with pytest.raises(ValueError):
        generate_binding_effects(
            total, signal_group_size, unaffected_lambda, affected_lambda
        )


def test_generate_perturbation_binding_data():
    # Setup
    gene_count = 100
    signal_group_size = 50
    gene_population = torch.randint(0, 2, (gene_count, 2), dtype=torch.int32)
    gene_population[:, 1] = (torch.arange(gene_count) < signal_group_size).int()

    # Call the function
    result = generate_perturbation_binding_data(gene_population)

    # Validate the result
    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert "gene_id" in result.columns, "DataFrame should have gene_id column"
    assert "signal" in result.columns, "DataFrame should have signal column"
    assert (
        "expression_effect" in result.columns
    ), "DataFrame should have expression_effect column"
    assert (
        "expression_pvalue" in result.columns
    ), "DataFrame should have expression_pvalue column"
    assert (
        "binding_effect" in result.columns
    ), "DataFrame should have binding_effect column"
    assert (
        "binding_pvalue" in result.columns
    ), "DataFrame should have binding_pvalue column"
    assert len(result) == gene_count, "DataFrame should have one row per gene"

    # Check data types
    assert pd.api.types.is_numeric_dtype(
        result["expression_effect"]
    ), "expression_effect should be numeric"
    assert pd.api.types.is_numeric_dtype(
        result["binding_effect"]
    ), "binding_effect should be numeric"
    assert pd.api.types.is_numeric_dtype(
        result["expression_pvalue"]
    ), "expression_pvalue should be numeric"
    assert pd.api.types.is_numeric_dtype(
        result["binding_pvalue"]
    ), "binding_pvalue should be numeric"
    assert pd.api.types.is_bool_dtype(result["signal"]), "signal should be boolean"
