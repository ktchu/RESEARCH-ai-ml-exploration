"""
The 'sparse' module provides utility functions for generating sparse datasets.
"""
# --- Imports

# External packages
import numpy as np


# --- Module functions

def generate_sparse_vectors(concept_space_dim, sparsity,
                            value_range, sample_size):
    """
    Generate dataset of sparse vectors in concept space. In each

    Parameters
    ----------
    concept_space_dim: int
        dimension of concept space

    sparsity: int
        number of nonzero components in each vector

    value_range: list-like
        allowed range for component values (inclusive)

    sample_size: int
        number of vectors to generate

    Return value
    ------------
    dataset: numpy.array
        Generated dataset. Each row of the array contains one vector in the
        dataset. dataset.shape = (sample_size, concept_space_dim)
    """
    # Check parameters
    if sparsity > concept_space_dim:
        raise ValueError("'sparsity' must be less than or equal to "
                         "'concept_space_dim'")

    if len(value_range) != 2:
        raise ValueError("'value_range' should have length 2")

    # Construct dataset
    dataset = np.zeros((sample_size, concept_space_dim))
    min_value = value_range[0]
    max_value = value_range[1]
    for i in range(sample_size):
        indices = np.random.choice(concept_space_dim, sparsity, replace=False)
        values = min_value \
            + (max_value - min_value) * np.random.random_sample(sparsity)
        dataset[i, indices] = values

    return dataset
