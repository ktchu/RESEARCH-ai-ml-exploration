#!/usr/bin/env python
# coding: utf-8

# # 2015-06-19: Using ICA to Learn Color Opponent Processing
# 
# *Last Updated*: 2020-11-29
# 
# ### Authors
# * Kevin Chu (kevin@velexi.com)
# 
# ### Overview
# In this Jupyter notebook, we demonstrate how color-opponent processing naturally arises as a way to decompose the RGB-based color space of natural scenes into statistically independent components. The Independent Component Analysis (ICA) algorithm is used to learn the statistically independent components.
# 
# Independent Component Analysis (ICA) identifies a collection of components/signals that(1) are statistically independent and (2) can be mixed (i.e., linearly combined) to obtain the observed signals. Mathematically, ICA uses observations $\{\mathbf{x}_1, \mathbf{x}_2, \ldots\}$ to estimate $m$ independent components $\mathbf{A}_j$ (each an $n$ dimensional vector) that can be linearly combined to construct the observations:
# $$
#   \mathbf{x}_i = \sum_{j=1}^{m} \mathbf{A}_j s_{ij}
# $$
# where $s_{ij}$ is the contribution of the $j$-th component to $\mathbf{x}_i$. In matrix form, this relationship can be expressed as:
# $$
#     \mathbf{x}_i = \mathbf{A} \mathbf{s}_i
# $$
# where $\mathbf{A}$ is the $n \times m$ _mixing matrix_ consisting of the $m$ independent components $\mathbf{A}_j$ as columns and $\mathbf{s}_i$ is the vector formed from $s_{ij}$.
# 
# For a particular observation $\mathbf{x}_i$, the contribution of each component can be computed by (1) inverting $\mathbf{A}$ if $\mathbf{A}$ is square or (2) solving the least squares problem if $\mathbf{A}$ is rectangular (with $m > n$).
# 
# ### Color Processing
# In the context of color processing, the _unmixing matrix_ (inverse of the mixing matrix) converts RGB values to three statistically independent components. The rows of this matrix define the computational processing that should be aplied to the raw RGB data to extract "color values" that are statistically independent. This example demonstrates that when applied to images from natural scenes, ICA indicates that RGB color values should be processed into "opponent colors" which is precisely how the human vision system processes raw color data.
# 
# ### Sensitivity of Results to Dataset Choice
# The color processing that is learned by ICA is sensitive to the choice of dataset. In particular, if the dataset is not representative of the colors that would be observed in the natural world, we should not expect that ICA would learn color opponent processing. Instead, ICA will learn a color processing scheme that yields statistically independent color values for the dataset used as input to the ICA algorithm.
# 
# ### User parameters
# 
# * `image_numbers`: image numbers to include in the analysis

# In[1]:


# --- Imports

# Standard library
import os
import random

# Third-party libraries
import numpy
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA


# In[2]:


# --- User parameters

# list of image numbers to include in the analysis
#
# Valid image numbers: 1 - 20
# Special values: "all"
image_numbers = [1, 5, 7, 10, 11, 14, 15, 20]
image_numbers = "all"


# ### Preparations

# In[3]:


# --- Verify user parameters

if image_numbers == "all":
    image_numbers = range(1, 21)
else:
    min_image_number = 1
    max_image_number = 20
    for item in image_numbers:
        try:
            item = int(item)
        except:
            message = "Image number '{}' is not an integer".format(item)
            raise ValueError(message)

        if item < min_image_number or item > max_image_number:
            message = "Image number '{}' is out of the range. Valid values: [{}, {}]" \
                .format(item, min_image_number, max_image_number)
            raise ValueError(message)


# In[4]:


# --- Load image data

jupyter_notebook = "2020-11-29-KTC-Using_ICA_to_Learn_Color_Opponent_Processing"
images = []
data_dir = os.path.join(os.path.abspath(os.path.dirname(jupyter_notebook)), "data")
for image_number in image_numbers:
    image_file = "natural_scene-{:02d}.jpg".format(image_number)
    image_path = os.path.join(data_dir, image_file)
    img = Image.open(image_path)
    images.append(numpy.array(img.getdata(), dtype='float'))


# In[5]:


# --- Construct data set

data = images[0]
for image in images[1:]:
    data = numpy.vstack((data, image))

# Scale color values to lie in the range [0, 1]
data = data / 255

# Apply gamma correction to get intensity
gamma = 2.2  # Reciprocal of typical gamma value applied when storing image
data = data ** gamma

# Apply log tranformation to get eye processing
data = numpy.log(data + 1)


# ### Perform ICA

# In[6]:


# --- Perform ICA

ica = FastICA(n_components=3)
ica.fit(data)

# --- Extract matrix that converts RGB to Color Opponent values

component_matrix = ica.components_
print(component_matrix)


# In[7]:


# --- Display results

# Extract color opponent processing coefficients
channels = {}
for i in range(3):
    # Get i-th row
    component = component_matrix[i, :]

    # Process channel
    if component[0]*component[1] < 0:
        # --- R-G channel
        
        # Make sure that R weight is positive
        if component[0] < 0:
            component *= -1

        channels["R-G Channel"] = component
    elif component[0]*component[1] > 0 and component[0]*component[2] < 0:
        # --- B-Y channel
        
        # Make sure that B weight is positive
        if component[2] < 0:
            component *= -1

        channels["B-Y Channel"] = component

    else:
        # --- Luminance channel
        
        # Make sure that luminance weights are all positive
        if component[0] < 0:
            component *= -1
        channels["Luminance Channel"] = component

# Output color opponent channels
print("ICA Components")
print("--------------")
print("Luminance Channel:", channels["Luminance Channel"])
print("R-G Channel:", channels["R-G Channel"])
print("B-Y Channel:", channels["B-Y Channel"])


# ### Observations
# 
# * __Luminance channel__. The luminosity is a sum of all of the RGB values.
# 
# * __R-G channel__. The red-green value is dominated by the difference between the red and green values with little contribution from the blue value.
# 
# * __B-Y channel__. The blue-yellow value is roughly equal to the difference between the blue value and the sum of the red and green values (which is effectively the color yellow).
