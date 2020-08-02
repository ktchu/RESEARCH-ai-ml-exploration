#!/usr/bin/env python
# coding: utf-8

# ## 2020-08-01: Exploring Sparsity Estimation Algorithms
# 
# *Last Updated*: 2020-08-02
# 
# ### Authors
# * Kevin Chu (kevin@velexi.com)
# 
# ### Overview
# In this Jupyter notebook, we explore algorithms for estimating dataset sparsity.
# 
# ### Definitions
# 
# * Let $\Omega$ be a union of $M$ linear subspaces $\Omega_i \subsetneq \mathbb{R}^n$ with $\dim \Omega_i \le s$: $\Omega = \bigcup_i^M \Omega_i$.
# 
# * Let $W$ be a dataset drawn from $\Omega$.
# 
# ### Key Results
# * Sampling of random sets of vectors from the dataset $W$ should be done _without replacement_. Since a set of vectors is linearly dependent if the set contains the same vector more than one time, sampling with replacement can lead to non-zero estimates for the probability that a sample of $k$ vectors is linearly dependent even if $k < \min( \dim \Omega_i )$.
# 
# ### User parameters
# 
# * `TODO`: TODO

# In[1]:


# --- Imports

# Standard library
import time

# External packages
import numba
import numpy as np
from numpy.linalg import qr
import seaborn as sns
import tqdm.notebook

# Local packages
from datasets.sparse import generate_sparse_vectors


# In[2]:


# --- User parameters

# Generative model parameters
concept_space_dim = 20
sparsity = 2
value_range = [0.5, 1.5]

# Dataset parameters
dataset_size = 10000

# Algorithm parameters
max_k = concept_space_dim
max_k = 6
sample_size = 100000


# In[3]:


# --- Define vectors_are_dependent() function

def vectors_are_dependent(vectors, tol=1e-12):
    """
    Determine whether a collection of vectors is linearly dependent.

    Parameters
    ----------
    vectors: numpy.ndarray
        set of vectors to determine linear dependence of. Note: it does not matter whether
        vectors are stored as rows or columns.
        
    Return value
    ------------
    dependent: bool
        True if vectors are linearly dependent; False otherwise
    
    min_abs_diag_R: float
        diagonal element of R with minimum absolute value
    """
    #  Handle edge case: 'vectors' contains a single vector
    if len(vectors.shape) == 1 or min(vectors.shape) == 1:
        min_abs_diag_R = np.abs(vectors).min()
        dependent = min_abs_diag_R < tol
        return dependent, min_abs_diag_R
    
    # Use the QR decomposition to transform the matrix 'vectors' into 
    # an upper-triangular matrix
    if vectors.shape[0] < vectors.shape[1]:
        R = qr(vectors.T, mode='r')
    else:
        R = qr(vectors, mode='r')

    # Compute diagonal element with the smallest absolute value
    if len(R.shape) == 1 or min(R.shape) == 1:
        # Case: R is a matrix with a single row or column
        min_abs_diag_R = abs(r[0])
    else:
        min_abs_diag_R = min(abs(np.diag(R)))

    # Determine if vectors are linearly dependent by comparing min_abs_diag_r to 0
    dependent = min_abs_diag_R < tol

    return dependent, min_abs_diag_R


# In[4]:


# --- Test vectors_are_dependent()

# ------ Exercise functionality and check results

# Dependent row vectors
vectors = np.array([[1,1,0],[2,2,0],[3,3,0]])
vectors = np.array([[1,1,0],[2,2,0]])
dependent, _ = vectors_are_dependent(vectors)
assert dependent

# Dependent column vectors
vectors = np.array([[1,2],[1,2],[0,0]])
dependent, _ = vectors_are_dependent(vectors)
assert dependent

# Independent column vectors
vectors = np.array([[1,1],[1,2],[0,0]])
dependent, _ = vectors_are_dependent(vectors)
assert not dependent

# Independent row vectors
vectors = np.array([[1,1,0],[1,2,0]])
dependent, _ = vectors_are_dependent(vectors)
assert not dependent

# Print test results
print("'vectors_are_dependent()' tests: PASSED")


# ### Preparations

# In[5]:


# --- Configuration

# Seaborn configuration
sns.set(color_codes=True)


# In[6]:


# Generate dataset
t_start = time.time()
dataset = generate_sparse_vectors(concept_space_dim, sparsity, value_range, dataset_size)
t_end = time.time()
time_generate_sparse_vectors = t_end - t_start

# Print timing data
print("Runtime 'generate_sparse_vectors()': {:.3g}s".format(time_generate_sparse_vectors))


# ### Summary of Significant Parameters

# In[7]:


# --- Display parameters that affect algorithm performance

print("s = {}".format(sparsity))
print("dataset size = {}".format(dataset_size))
print("sample size = {}".format(sample_size))


# ### Probability that $k$ vector samples are linearly dependent

# #### Sample vectors with replacement
# 
# * Sampling with replacement is not a good idea. For a finite dataset $W$ (in contrast to the   space $\Omega$), there is a nonzero probability of drawing a sample that is linearly dependent just because the same vecor from $W$ is drawn more than one time.

# In[8]:


# --- Compute probabilities when samples are taken with replacement

# Initialize probabilities that outer products of $k$ vectors is zero
p_dependent_with_replacement = np.zeros([max_k])
    
# Loop over number of vectors to sample from dataset

with tqdm.notebook.trange(2, max_k+1) as k_range:
    for k in k_range:
        k_range.set_description("Processing k={} (with replacement)".format(k))
        
        # Initialize count for linearly dependent samples
        count_dependent = 0
    
        # Generate samples and check linear dependence
        for _ in tqdm.notebook.tqdm(range(sample_size), unit='samples',
                                    desc='Sampling {} vectors'.format(k), leave=False):
            indices = np.random.choice(dataset_size, k, replace=True)
            vectors = dataset[indices, :]
            dependent, min_abs_diag_R = vectors_are_dependent(vectors)
            if dependent:
                count_dependent += 1
            
            p_dependent_with_replacement[k-1] = count_dependent / sample_size

# Display results
print("Probabilities when samples are taken with replacement")
for k in range(max_k):
    print('P(outer product of {} vectors = 0) = {:.5g}'
          .format(k+1, p_dependent_with_replacement[k]))


# #### Sample vectors without replacement

# In[9]:


# --- Compute probabilities when samples are taken without replacement

# Initialize probabilities that outer products of $k$ vectors is zero
p_dependent_without_replacement = np.zeros([max_k])
    
# Loop over number of vectors to sample from dataset

with tqdm.notebook.trange(2, max_k+1) as k_range:
    for k in k_range:
        k_range.set_description("Processing k={} (without replacement)".format(k))
        
        # Initialize count for linearly dependent samples
        count_dependent = 0
    
        # Generate samples and check linear dependence
        for _ in tqdm.notebook.tqdm(range(sample_size), unit='samples',
                                    desc='Sampling {} vectors'.format(k), leave=False):
            indices = np.random.choice(dataset_size, k, replace=False)
            vectors = dataset[indices, :]
            dependent, min_abs_diag_R = vectors_are_dependent(vectors)
            if dependent:
                count_dependent += 1
            
            p_dependent_without_replacement[k-1] = count_dependent / sample_size

# Display results
print("Probabilities when samples are taken without replacement")
for k in range(max_k):
    print('P(outer product of {} vectors = 0) = {:.5g}'
          .format(k+1, p_dependent_without_replacement[k]))

