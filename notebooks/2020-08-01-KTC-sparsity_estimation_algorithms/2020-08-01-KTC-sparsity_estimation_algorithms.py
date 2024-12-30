#!/usr/bin/env python
# coding: utf-8

# # 2020-08-01: Exploring Sparsity Estimation Algorithms
# 
# *Last Updated*: 2020-08-03
# 
# ### Authors
# * Kevin Chu (kevin@velexi.com)
# 
# ### Overview
# In this Jupyter notebook, we explore algorithms for estimating dataset sparsity based on the probability that a sample of $k$ vectors from a dataset is linearly dependent.
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
import math
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
concept_space_dim = 5
sparsity = 2
value_range = [0.5, 1.5]

# Dataset parameters
dataset_size = 20000

# Algorithm parameters
max_k = concept_space_dim
max_k = 6
sample_size = 10000


# ### Helper Functions

# #### Linear Dependence of Sets of Vectors

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


# In[ ]:


# --- Define compute_P_vectors_are_dependent() function

# @numba.jit(nopython=True, nogil=True)
def compute_P_vectors_are_dependent(num_subspaces, sparsity, r):
    """
    Compute probability that a sample of 'r' vectors drawn with replacement from
    'num_subspaces' subspaces with the specified sparsity is linearly dependent.
    
    Parameters
    ----------
    num_subspaces: int
        number of subspaces

    sparsity: int
        sparsity of subspaces

    r: int
        number of vectors to sample

    Return value
    ------------
    probability: float
        probability that a sample of 'r' vectors drawn with replacement from
        'num_subspaces' subspaces with the specified sparsity is linearly dependent.
    """
    # --- Check parameters
    
    if num_subspaces < 1:
        raise ValueError("'num_subspaces' must be positive")
    if sparsity < 1:
        raise ValueError("'sparsity' must be positive")
    if r < 1:
        raise ValueError("'r' must be positive")
        
    # --- Handle edge cases

    if r < sparsity + 1:
        return 0

    # --- Compute probability

    p_one_object = 0
    for i in range(p, m+1):
        p_one_object += math.comb(m, i) * (1/n)**i * (1 - 1/n)**(m-i)
    probability = n * p_one_object

    return probability

# --- Test compute_P()

# ------ Test Case #1

# Preparations
n_test = 5
m_test = 3
p_test = 1

# Exercise functionality
t_start = time.time()
probability_test = compute_P(n_test, m_test, p_test)
t_end = time.time()
time_compute_P_first_call = t_end - t_start

# Check results
print(probability_test)

# Verify computational performance boost from Numba
t_start = time.time()
probability_test = compute_P(n_test, m_test, p_test)
t_end = time.time()
time_compute_P_second_call = t_end - t_start

# ------ Test Case #1

# Preparations
n_test = 10
m_test = 4
p_test = 3

# Exercise functionality
t_start = time.time()
probability_test = compute_P(n_test, m_test, p_test)
t_end = time.time()

# Check results
print(probability_test)

# ------ Print results

# TODO
# print("'count_p_samples()' tests: PASSED")
print("Runtime 'compute_P()' (with compilation): {:.3g}s"
      .format(time_compute_P_first_call))

print("Runtime 'compute_P()' (after compilation): {:.3g}s"
      .format(time_compute_P_second_call))


# #### Computing number of arrangements with at least one object selected at least $p$ times.

# #### Definitions
# 
# * Let $A_{n, m, p}$ denote the number of $m$-tuples where
# 
#     * each element of the tuple is drawn from a pool of $n$ classes and
# 
#     * at least one class is represented at least $p$ times in the $m$-tuple.

# In[5]:


# --- Define count_p_samples()

# @numba.jit(nopython=True, nogil=True)
def compute_A_direct_count(n, m, p):
    """
    Count number of samples of size 'm' drawn with replacement from 'n' objects with at
    least one object represented at least 'p' times.
    
    This function computes A by explicitly counting them.

    Parameters
    ----------
    n: int
        number of objects

    m: int
        number of samples to draw

    p: int
        target number of times at least one object is represented in the sample

    Return value
    ------------
    num_p_samples: int
        number of samples of size 'm' that satisfies the property that at least one object
        is represented at least 'p' times
    """
    # --- Check parameters
    
    if n < 1:
        raise ValueError("'n' must be positive")

    if m < 1:
        raise ValueError("'m' must be positive")

    if p < 1:
        raise ValueError("'p' must be positive")

    # --- Preparations

    # Initialize count
    count = 0

    # --- Count number of 

    for i in range(1, n+1):
        for j in range(1, min(i+1, m+1)):
            for k in range(1, min(j+1, p+1)):
                print(i, j, k)
                if k == j:
                    sub_problem_solutions[(i, j, k)] = i
                elif k > j:
                    sub_problem_solutions[(i, j, k)] = 0
                else:
                    num_k_samples = 0
                    for t in range(0, k):
                        if i-1 >= j-t and j-t >= k:
                            num_k_samples +=                                 math.comb(j, t) * sub_problem_solutions[(i-1, j-t, k)]
                    for t in range(k, j+1):
                        num_k_samples += math.comb(j, t) * (i-1)**(j-t)
                    

                    sub_problem_solutions[(i, j, k)] = num_k_samples
                    
    num_p_samples = sub_problem_solutions[n, m, p]
#    print(len(sub_problem_solutions))
#    print(sub_problem_solutions)

    return num_p_samples


# ##### Analytical Formulas
# 
# ##### Formula based on recursive formula
# 
# * $A_{n, m, p}$ satisfies the recurrence relation:
# 
#     $$
#     A_{n, m, p} = \sum_{i = 0}^{p-1} {m \choose i} A_{n-1, m-i, p}
#                 + \sum_{i = p}^m {m \choose i} (n-1)^{m-i}
#     $$
# 
# * __TODO__. Need to fix these formulas

# In[6]:


# --- Define count_p_samples()

# @numba.jit(nopython=True, nogil=True)
def compute_A(n, m, p):
    """
    Count number of samples of size 'm' drawn with replacement from 'n' objects with at
    least one object represented at least 'p' times.
    
    We use dynamic programming to solve this problem.
    
    Parameters
    ----------
    n: int
        number of objects

    m: int
        number of samples to draw

    p: int
        target number of times at least one object is represented in the sample

    Return value
    ------------
    num_p_samples: int
        number of samples of size 'm' that satisfies the property that at least one object
        is represented at least 'p' times
    """
    # --- Check parameters
    
    if n < 1:
        raise ValueError("'n' must be positive")
        
    # --- Handle edge cases

    if n < m:
        return 0
    elif m < p:
        return 0
    elif m == p:
        return n

    # --- Preparations

    # Initialize dynamic programming "table"
    sub_problem_solutions = {}

    # Initialize num_p_samples
    num_p_samples = 0

    # --- Compute num_p_samples using dynamic programming

    for i in range(1, n+1):
        for j in range(1, min(i+1, m+1)):
            for k in range(1, min(j+1, p+1)):
                print(i, j, k)
                if k == j:
                    sub_problem_solutions[(i, j, k)] = i
                elif k > j:
                    sub_problem_solutions[(i, j, k)] = 0
                else:
                    num_k_samples = 0
                    for t in range(0, k):
                        if i-1 >= j-t and j-t >= k:
                            num_k_samples +=                                 math.comb(j, t) * sub_problem_solutions[(i-1, j-t, k)]
                    for t in range(k, j+1):
                        num_k_samples += math.comb(j, t) * (i-1)**(j-t)
                    

#                     sub_problem_solutions[(i, j, k)] = num_k_samples
                    
    num_p_samples = sub_problem_solutions[n, m, p]
#    print(len(sub_problem_solutions))
#    print(sub_problem_solutions)

    return num_p_samples

# # --- Test count_p_samples()

# # ------ Test Case #1

# # Preparations
# n_test = 5
# m_test = 3
# p_test = 1

# # Exercise functionality
# t_start = time.time()
# num_p_samples_test = count_p_samples(n_test, m_test, p_test)
# t_end = time.time()
# time_count_p_samples_first_call = t_end - t_start

# # Check results
# print(num_p_samples_test)

# # Verify computational performance boost from Numba
# t_start = time.time()
# num_p_samples_test = count_p_samples(n_test, m_test, p_test)
# t_end = time.time()
# time_count_p_samples_second_call = t_end - t_start

# # ------ Test Case #2

# # Preparations
# n_test = 10
# m_test = 1
# p_test = 2

# # Exercise functionality
# t_start = time.time()
# num_p_samples_test = count_p_samples(n_test, m_test, p_test)
# t_end = time.time()
# time_count_p_samples = t_end - t_start

# # Check results
# assert num_p_samples_test == 0

# # ------ Test Case #3

# # Preparations
# n_test = 10
# m_test = 12
# p_test = 2

# # Exercise functionality
# t_start = time.time()
# num_p_samples_test = count_p_samples(n_test, m_test, p_test)
# t_end = time.time()
# time_count_p_samples = t_end - t_start

# # Check results
# assert num_p_samples_test == 0

# # ------ Test Case #4

# # Preparations
# n_test = 10
# m_test = 4
# p_test = 4

# # Exercise functionality
# t_start = time.time()
# num_p_samples_test = count_p_samples(n_test, m_test, p_test)
# t_end = time.time()
# time_count_p_samples = t_end - t_start

# # Check results
# assert num_p_samples_test == n_test

# # ------ Test Case #5

# # Preparations
# n_test = 10
# m_test = 6
# p_test = 2

# # Exercise functionality
# t_start = time.time()
# num_p_samples_test = count_p_samples(n_test, m_test, p_test)
# t_end = time.time()
# time_count_p_samples = t_end - t_start

# # ------ Test Case #6

# # Preparations
# n_test = 10
# m_test = 3
# p_test = 3

# # Exercise functionality
# t_start = time.time()
# num_p_samples_test = count_p_samples(n_test, m_test, p_test)
# t_end = time.time()
# time_count_p_samples = t_end - t_start

# # Check results
# print(num_p_samples_test)

# # ------ Print results

# # TODO
# # print("'count_p_samples()' tests: PASSED")
# print("Runtime 'count_p_samples()' (with compilation): {:.3g}s"
#       .format(time_count_p_samples_first_call))

# print("Runtime 'count_p_samples()' (after compilation): {:.3g}s"
#       .format(time_count_p_samples_second_call))


# ### Preparations

# In[8]:


# --- Configuration

# Seaborn configuration
sns.set(color_codes=True)


# In[9]:


# Compute number of subspaces
num_subspaces = math.comb(concept_space_dim, sparsity)

# Generate dataset
t_start = time.time()
dataset = generate_sparse_vectors(concept_space_dim, sparsity, value_range, dataset_size)
t_end = time.time()
time_generate_sparse_vectors = t_end - t_start

# Print timing data
print("Runtime 'generate_sparse_vectors()': {:.3g}s".format(time_generate_sparse_vectors))


# ### Summary of Significant Parameters

# In[10]:


# --- Display parameters that affect algorithm performance

print("concept space dimension = {}".format(concept_space_dim))
print("sparsity = {}".format(sparsity))
print("num_subspaces = {}".format(num_subspaces))

print("dataset size = {}".format(dataset_size))
print("sample size = {}".format(sample_size))


# In[11]:


# # --- Formula based on recursive formula

# print("Number of linearly dependent samples")
# for k in range(1, max_k+1):
#     N = count_p_samples(num_subspaces, k, sparsity+1)
#     print('    N({} sample vectors) = {}'.format(k, N))
#     print('    P(outer product of {} vectors = 0) = {}'.format(k, N/num_subspaces**k))
    

# for k in range(1, max_k+1):
#     count = 0
#     for s in range(1, k+1):
#         print('AA', num_subspaces, k, s, count_p_samples(num_subspaces, k, s))
#         count += count_p_samples(num_subspaces, k, s)

#     print('k = {}'.format(k))
#     print('total number of arrangements({}) = {}'.format(k, count))
#     print('num subspaces**k = {}'.format(num_subspaces**k))


# #### Formula based on counting arrangements of boundaries, grouped vectors, and individual vectors
# 
# * TODO: clean up definitions
# 
# * The space between the $i$-th and $(i+1)$-th boundaries represents the vectors contained in the $i$-th subspace. The position of the first and last boundaries are fixed.
# * A group of $s+1$ vectors that are linearly dependent (i.e., all reside in a single subspace) is treated as a single object.
# * The remaining $k - (s+1)$ vectors are treated as individual objects.
# 
# ##### Open Questions
# * Do we need a factor of $(s+1)!$ is to account for the fact that there are that many different arrangements of the vectors in the linearly dependent set?

# In[12]:


# --- Formula based on arrangement counting

print("Number of linearly dependent sample (via arrangement counting)")
for k in range(1, max_k+1):
    num_individual_vectors = k - (sparsity + 1)
    num_objs = (num_subspaces - 1) + 1 + num_individual_vectors

    if num_individual_vectors == 0:
        N = math.comb(num_objs, 1)
    elif num_individual_vectors > 0:
        N = math.comb(num_objs, 1) * math.comb(num_objs-1, num_individual_vectors)             * math.factorial(num_individual_vectors) * math.factorial(sparsity + 1)
    else:
        N = 0

    print('    N({} sample vectors) = {}'.format(k, N))
    print('    P(outer product of {} vectors = 0) = {}'.format(k, N/num_subspaces**k))

for k in range(1, max_k+1):
    test_N = 0
    for j in range(1, min(k+1, num_subspaces+1)):
        num_individual_vectors = k - j
        num_objs = (num_subspaces - 2) + 1 + num_individual_vectors

        print(num_subspaces + k - 2, j, num_objs, num_individual_vectors)
        if num_individual_vectors == 0:
            test_N += math.comb(num_objs, 1)
        elif num_individual_vectors > 0:
            test_N += math.comb(num_objs, 1) * math.comb(num_objs-1, num_individual_vectors)
#                * math.factorial(num_individual_vectors) * math.factorial(test_sparsity + 1)
        print(test_N)

    print('k = {}'.format(k))
    print('test_N({}) = {}'.format(k, test_N))
    print('num movable boundaries + num vectors = {}'.format(num_subspaces + k - 2))
    print('(num movable boundaries + num vectors) choose (num movable boundaries) = {}'
          .format(math.comb(num_subspaces + k - 2, num_subspaces - 2)))


# ### Empirical Probabilities

# #### Sample vectors with replacement
# 
# * _False Positives_. For a finite dataset $W$ (in contrast to the   space $\Omega$), there
#   is a nonzero probability of drawing a sample that is linearly dependent just because the
#   same vector from $W$ is drawn more than one time.

# In[13]:


# # --- Compute probabilities when samples are taken with replacement

# # Initialize probabilities that outer products of $k$ vectors is zero
# p_dependent_with_replacement = np.zeros([max_k])
    
# # Loop over number of vectors to sample from dataset

# with tqdm.notebook.trange(2, max_k+1) as k_range:
#     for k in k_range:
#         k_range.set_description("Processing k={} (with replacement)".format(k))
        
#         # Initialize count for linearly dependent samples
#         count_dependent = 0
    
#         # Generate samples and check linear dependence
#         for _ in tqdm.notebook.tqdm(range(sample_size), unit='samples',
#                                     desc='Sampling {} vectors'.format(k), leave=False):
#             indices = np.random.choice(dataset_size, k, replace=True)
#             vectors = dataset[indices, :]
#             dependent, min_abs_diag_R = vectors_are_dependent(vectors)
#             if dependent:
#                 count_dependent += 1
            
#             p_dependent_with_replacement[k-1] = count_dependent / sample_size

# # Display results
# print("Probabilities when samples are taken with replacement")
# for k in range(max_k):
#     print('    P(outer product of {} vectors = 0) = {:.5g}'
#           .format(k+1, p_dependent_with_replacement[k]))


# #### Sample vectors without replacement
# 
# * _Poor Computational Performance_. The computational performance of sampling without
#   replacement is much lower than sampling with replacement.

# In[15]:


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
    expected_probability = compute_P(num_subspaces, k+1, sparsity+1)
    print('    P(outer product of {} vectors = 0) = {:.5g}, Expected = {:0.5g}'
          .format(k+1, p_dependent_without_replacement[k], expected_probability))

