"""
Tests for 'sparse' module.
"""
# --- Imports

# External packages
import numpy as np

# Local packages
from datasets.sparse import generate_sparse_vectors


# --- Tests

def test_generate_sparse_vectors():
    """
    Basic tests for generate_sparse_vectors().
    """
    # --- Preparations

    concept_space_dim = 10
    sparsity = 3
    value_range = [0.1, 1.0]
    sample_size = 100

    # --- Exercise functionality

    dataset = generate_sparse_vectors(concept_space_dim, sparsity,
                                      value_range, sample_size)

    # --- Check results

    # Check dataset shape
    assert dataset.shape == (sample_size, concept_space_dim)

    # Check sparsity
    for i in range(sample_size):
        assert np.count_nonzero(dataset[i]) == sparsity

    # Check component values
    assert dataset.max() <= value_range[1]
    assert dataset[np.where(dataset != 0)].min() >= value_range[0]
