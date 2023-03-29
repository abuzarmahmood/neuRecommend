from itertools import combinations
import numpy as np
import pylab as plt
from joblib import load
from umap.parametric_umap import load_ParametricUMAP
import json
import os
from glob import glob
import sys
sys.path.append('../create_pipeline')
from feature_engineering_pipeline import *
from return_data import return_data

params_dir = '../create_pipeline/params'
with open(os.path.join(params_dir, 'path_vars.json'), 'r') as path_file:
    path_vars = json.load(path_file)
with open(os.path.join(params_dir, 'data_params.json'), 'r') as path_file:
    data_params = json.load(path_file)
zero_ind = data_params['zero_ind']

############################################################
# Load data and pipelines
############################################################

X_raw, y = return_data()

# Load feature engineering Pipeline
fe_pipeline = load(os.path.join(path_vars['feature_pipeline_path']))

# Load UMAP model
umap_model_paths = glob(os.path.join(path_vars['model_save_dir'], 'umap_model*')) 
umap_models = [load_ParametricUMAP(x) for x in umap_model_paths]

# Split data by polarity
polarity = np.sign(AmpFeature(zero_ind).transform(X_raw)).flatten()

unique_polarity = np.unique(polarity)
sorted_model_paths = [[x for x in umap_model_paths if "_"+str(int(this_polarity)) in x] \
        for this_polarity in unique_polarity]
sorted_model_paths = np.array(sorted_model_paths).flatten() 
umap_models = [load_ParametricUMAP(x) for x in sorted_model_paths]

X = fe_pipeline.transform(X_raw)
polar_data = [X[polarity == this_polarity] for this_polarity in unique_polarity]
polar_y = [y[polarity == this_polarity] for this_polarity in unique_polarity]

############################################################
# Run pipeline
############################################################
X_umap = [this_model.transform(this_data) for this_model, this_data in zip(umap_models, polar_data)]

############################################################
# Make Plots
############################################################
# Make subplots of all combinations of first 3 dimensions as 2D scatter plots
thinning = 100
cmap = plt.cm.get_cmap('tab10')
plot_dat = [this_data[::thinning] for this_data in X_umap]
c = [cmap(this_y[::thinning]) for this_y in polar_y]

# Plot scatters and histograms for each element in plot_dat
fig, ax = plt.subplots(len(plot_dat), 2, figsize=(7, 7), sharex='row', sharey='row')
for num, this_plot_dat in enumerate(plot_dat):
    this_y = polar_y[num]
    scatter = ax[num,0].scatter(
        this_plot_dat[:, 0],
        this_plot_dat[:, 1],
        alpha=0.1,
        c=c[num])
    ax[num,0].set_xlabel('dim0')
    ax[num,0].set_ylabel('dim1')
    plt.colorbar(scatter, ax=ax[num,0])
    #this_ax.legend(
    #    handles=scatter.legend_elements()[0],
    #    labels=['False', 'True'],
    #)
    im = ax[num,1].hist2d(*this_plot_dat.T, bins=40,
                     cmap=plt.cm.get_cmap('Greys'))
    plt.colorbar(im[3], ax=ax[num,1])
plt.show()

#fig, ax = plt.subplots(1, 3, figsize=(10, 5))
#inds = list(combinations(np.arange(3), 2))
#for num, (this_ind1, this_ind2) in enumerate(inds):
#    scatter = ax[num].scatter(
#        plot_dat[:, this_ind1],
#        plot_dat[:, this_ind2],
#        alpha=0.1,
#        c=c)
#    ax[num].set_xlabel('dim' + str(this_ind1))
#    ax[num].set_ylabel('dim' + str(this_ind2))
#    ax[num].legend(
#        handles=scatter.legend_elements()[0],
#        labels=['False', 'True'],
#    )
#plt.show()
