"""
Testing whether a new sample significantly increases the entropy of the 
distribution

Dealing with high-dimensional (probably 5-10) datapoints

Refer to:
    https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
    https://stats.stackexchange.com/questions/346137/how-to-compute-joint-entropy-of-high-dimensional-data
"""

import numpy as np
import scipy as sp
import pylab as plt
from tqdm import tqdm, trange

def entropy(x, bins = None):
    if bins is None:
        counts = np.histogramdd(x)[0]
    else:
        counts = np.histogramdd(x, bins=bins)[0]
    dist = counts / np.sum(counts)
    logs = np.log2(np.where(dist > 0, dist, 1))
    return -np.sum(dist * logs)

############################################################
## Test 1
############################################################
# How many samples before a parametric distribution is "well-defined"?

# 1D Uniform
bins = [np.linspace(0,6,10)]
max_samples = 1000
dataset = []
entropy_list = []
for i in trange(max_samples):
    x = np.random.random()
    dataset.append(x)
    this_entropy = entropy(dataset, bins = bins)
    entropy_list.append(this_entropy)

# Start adding different samples
for i in trange(max_samples):
    x = np.random.random()+5
    dataset.append(x)
    this_entropy = entropy(dataset, bins = bins)
    entropy_list.append(this_entropy)

plt.plot(np.arange(len(entropy_list)), entropy_list, '-x'); 
plt.xlabel('Number of samples')
plt.ylabel('Distribution Entropy')
plt.title('Uniform Distribution')
plt.show()

# 3D Uniform
bins = [np.linspace(0,6,10)]*3
max_samples = 1000
dataset = [] 
entropy_list = []
for i in trange(max_samples):
    x = np.random.random(3)
    dataset.append(x)
    this_entropy = entropy(np.stack(dataset), bins = bins)
    entropy_list.append(this_entropy)

# Start adding different samples
for i in trange(max_samples):
    x = np.random.random(3)+5
    dataset.append(x)
    this_entropy = entropy(np.stack(dataset), bins = bins)
    entropy_list.append(this_entropy)

plt.plot(np.arange(len(entropy_list)), entropy_list, '-x'); 
plt.xlabel('Number of samples')
plt.ylabel('Distribution Entropy')
plt.title('Uniform Distribution')
plt.show()
