#!/usr/bin/env python
"""
Experiment that demonstrates how color-opponent processing naturally arises as
a way to decompose the RGB-based color space of natural scenes into
statistically independent components.
"""

# --- Imports

# Standard library
import os
import random

# Third-party libraries
import numpy
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# --- Load data

# load natural image data
image_file_list = os.listdir(os.path.join('data'))
image_data = []
for image_file in image_file_list:
    img = Image.open(os.path.join('data', image_file))
    image_data.append(numpy.array(img.getdata(), dtype='float'))

# --- Construct data set

data = image_data[0]
for data_set in image_data[1:]:
    data = numpy.vstack((data, data_set))

# Scale color values to lie in the range [0, 1]
data = data / 255

# Apply gamma correction to get intensity
gamma = 2.2  # Reciprocal of typical gamma value applied when storing image
data = data ** gamma

# Apply log tranformation to get eye processing
data = numpy.log(data + 1)

# --- Perform ICA

ica = FastICA(n_components=3)
ica.fit(data)

# --- Extract source components

components = ica.components_

# --- Output results

print("ICA Components")
print("--------------")
channels = {}
for component in components:
    if component[0]*component[1] < 0:
        # Make sure that R weight is positive
        if component[0] < 0:
            component *= -1

        channels['R-G Channel'] = component
    elif component[0]*component[1] > 0 and component[0]*component[2] < 0:
        # Make sure that B weight is positive
        if component[2] < 0:
            component *= -1

        channels['B-Y Channel'] = component

    else:
        # Make sure that luminance weights are all positive
        if component[0] < 0:
            component *= -1
        channels['Luminance Channel'] = component

print('Luminance Channel:', channels['Luminance Channel'])
print('R-G Channel:', channels['R-G Channel'])
print('B-Y Channel:', channels['B-Y Channel'])
print()

# --- Plot results

# print("Data")
# sample = random.sample(range(data.shape[0]), 10)
# data_sample = data[sample, :]
# print(data_sample)
# print()
#
# print("Transformed Data")
# transformed_data = ica.transform(data_sample)
# print(transformed_data)
# print()
#
# num_plot_points = 100000
# transformed_data = ica.transform(data[0:num_plot_points, :])
# plt.figure(1)
# plt.scatter(data[0:num_plot_points, 0], data[0:num_plot_points, 1])
#
# plt.figure(2)
# plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
#
# plt.show()
