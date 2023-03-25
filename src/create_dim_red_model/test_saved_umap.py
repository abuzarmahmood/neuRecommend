import sys
sys.path.append('../create_pipeline')

from return_data import return_data
import os
import json
from umap.parametric_umap import load_ParametricUMAP
from joblib import load
import pylab as plt
import numpy as np
from itertools import combinations

with open('./params/path_vars.json', 'r') as path_file:
    path_vars = json.load(path_file)

############################################################
# Load data and pipelines 
############################################################

X_raw, y = return_data()

# Load feature engineering Pipeline
fe_pipeline = load(os.path.join(
    path_vars['model_save_dir'], 'umap_feature_pipeline.dump')) 

# Load UMAP model
umap_model = load_ParametricUMAP(os.path.join(
    path_vars['model_save_dir'], 'umap_model'))

############################################################
# Run pipeline
############################################################
X = fe_pipeline.transform(X_raw)
X_umap = umap_model.transform(X)

############################################################
# Make Plots
############################################################
# Make subplots of all combinations of first 3 dimensions as 2D scatter plots
thinning = 100
cmap = plt.cm.get_cmap('tab10')
plot_dat = X_umap[::thinning]
c = cmap(y[::thinning])
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
inds = list(combinations(np.arange(3), 2))
for num, (this_ind1, this_ind2) in enumerate(inds):
    scatter = ax[num].scatter(
        plot_dat[:, this_ind1],
        plot_dat[:, this_ind2],
        alpha = 0.1,
        c=c)
    ax[num].set_xlabel('dim' + str(this_ind1))
    ax[num].set_ylabel('dim' + str(this_ind2))
    ax[num].legend(
        handles=scatter.legend_elements()[0],
        labels=['False', 'True'],
    )
plt.show()
