# mypy: disable-error-code=arg-type
import pytest
import torch

from yeastdnnexplorer.probability_models.generate_data import (
    GenePopulation,
    generate_binding_effects,
    generate_gene_population,
    generate_perturbation_effects,
    generate_pvalues,
)


def test_generate_gene_population():
    total_genes = 1000
    bound_ratio = 0.3
    bound_group_size = int(total_genes * bound_ratio)

    gene_population = generate_gene_population(total_genes, bound_ratio)

    # Check if the output is a 1D tensor
    assert gene_population.labels.ndim == 1

    # Check if the output has the correct shape
    assert gene_population.labels.shape == torch.Size([total_genes])

    # Check if the second column contains the correct number of bound
    # and non-bound genes
    assert torch.sum(gene_population.labels) == bound_group_size
    assert torch.sum(gene_population.labels == 0) == total_genes - bound_group_size

    # Additional tests could include checking the datatype of the tensor elements
    assert gene_population.labels.dtype == torch.bool


def test_generate_binding_effects_success():
    # set torch seed
    torch.manual_seed(42)
    # Create a mock GenePopulation with some genes
    # labeled as bound and others as unbound
    gene_population = GenePopulation(torch.tensor([1, 0, 1, 0], dtype=torch.bool))
    # Call generate_binding_effects with valid arguments
    enrichment = generate_binding_effects(gene_population)
    # Check that the result is a tensor of the correct shape
    assert isinstance(enrichment, torch.Tensor)
    assert enrichment.shape == (4,)


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


def test_generate_perturbation_effects_with_and_without_adjustment():
    torch.manual_seed(42)
    # Create mock binding data with the first
    # column indicating bound (1) or unbound (0),
    # the second column indicates the enrichment, and the third the p-value.
    # Add an extra dimension for TFs -- the function requires a 3D tensor.
    binding_data = torch.tensor(
        [
            [1.0000, 0.5000, 0.0700],
            [0.0000, 0.2000, 0.0500],
            [1.0000, 0.8000, 0.0100],
            [0.0000, 0.1000, 0.9000],
        ]
    ).unsqueeze(
        1
    )  # Add TF dimension

    # Specify means and standard deviations
    unbound_mean = 0.0
    unbound_std = 1.0
    bound_mean = 4.0
    bound_std = 1.0

    # First, test without mean adjustment
    effects_without_adjustment = generate_perturbation_effects(
        binding_data=binding_data,
        tf_index=0,
        unbound_mean=unbound_mean,
        unbound_std=unbound_std,
        bound_mean=bound_mean,
        bound_std=bound_std,
        max_mean_adjustment=0.0,  # No adjustment
    )

    # Extract masks for bound and unbound genes based on labels
    bound_mask = binding_data[:, :, 0].squeeze() == 1
    unbound_mask = binding_data[:, :, 0].squeeze() == 0

    # Assert the effects tensor is of the correct shape
    assert effects_without_adjustment.shape[0] == binding_data.shape[0]

    assert torch.isclose(
        torch.abs(effects_without_adjustment[bound_mask]).mean(),
        torch.tensor(bound_mean),
        atol=bound_std,
    )
    assert torch.isclose(
        torch.abs(effects_without_adjustment[~bound_mask]).mean(),
        torch.tensor(unbound_mean),
        atol=unbound_std,
    )
    assert torch.isclose(
        torch.abs(effects_without_adjustment[bound_mask]).std(),
        torch.tensor(bound_std),
        atol=bound_std,
    )
    assert torch.isclose(
        torch.abs(effects_without_adjustment[~bound_mask]).std(),
        torch.tensor(unbound_std),
        atol=unbound_std,
    )

    # Test with mean adjustment
    effects_with_adjustment = generate_perturbation_effects(
        binding_data=binding_data,
        tf_index=0,
        unbound_mean=unbound_mean,
        unbound_std=unbound_std,
        bound_mean=bound_mean,
        bound_std=bound_std,
        max_mean_adjustment=4.0,  # Applying adjustment
    )

    # Assert that bound genes with adjustments have a mean effect greater than
    # the base mean
    assert (
        torch.abs(effects_with_adjustment[bound_mask]).mean()
        > torch.abs(effects_without_adjustment[bound_mask]).mean()
    )

    # Assert that the mean effect for unbound genes remains close to the unbound mean
    assert torch.isclose(
        torch.abs(effects_with_adjustment[unbound_mask]).mean(),
        torch.tensor(unbound_mean),
        atol=unbound_std,
    )
    # and that the unbound standard deviation remains close to the unbound std
    assert torch.isclose(
        torch.abs(effects_with_adjustment[unbound_mask]).std(),
        torch.tensor(unbound_std),
        atol=unbound_std,
    )
