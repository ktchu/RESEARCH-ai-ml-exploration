#!/usr/bin/env python
"""
Experiment that demonstrates how projections onto Gabor filters naturally
arise as a way to decompose the spatial variations in natural scenes into
statistically independent components.

Notes
-----
* The original version of this script was written by Andreas Mueller.

"""

# --- Imports

# Standard library
import os

# Third-party libraries
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# --- Load data

# load natural image patches
# NOTE: original data can be loaded from mldata.org by using
#       fetch_mldata("natural scenes data")
data_frame = pandas.DataFrame.from_csv(
    os.path.join("data", "natural-scenes-data.csv"),
    header=None, index_col=None)

# Reshape original 1000 patches of size 32x32 to 16000 patches of size 8x8
data = data_frame.as_matrix().reshape(1000, 4, 8, 4, 8)
data = numpy.rollaxis(data, 3, 2).reshape(-1, 8 * 8)

# --- Perform ICA

ica = FastICA(n_components=49)
ica.fit(data)
filters = ica.components_

# --- Plot results

for i, f in enumerate(filters):
    plt.subplot(7, 7, i + 1)
    plt.imshow(f.reshape(8, 8), cmap="gray")
    plt.axis("off")

plt.show()
