from sklearn.mixture import BayesianGaussianMixture
import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy as shc
import os

import sys
pipeline_dir = '/media/bigdata/projects/neuRecommend/src/create_pipeline'
sys.path.append(pipeline_dir)
from feature_engineering_pipeline import *
from return_data import return_data

params_dir = os.path.join(pipeline_dir, 'params')
with open(os.path.join(params_dir, 'path_vars.json'), 'r') as path_file:
    path_vars = json.load(path_file)

random_state = 3
rng = np.random.RandomState(random_state)

############################################################
# Load Data
############################################################

fin_data, fin_labels, fin_groups = return_data()
fin_groups = np.vectorize(int)(fin_groups)
ind_frame = pd.DataFrame(
    dict(
        labels=fin_labels,
        groups=fin_groups,
    )
)
ind_frame['cat_ind'] = ind_frame['labels'].astype(str) + \
    '_' + ind_frame['groups'].astype(str)
ind_frame['cat_ind_fin'] = ind_frame['cat_ind'].astype('category').cat.codes

# Only process spikes for now 
ind_frame = ind_frame[ind_frame['labels'] == 1]

############################################################
# Select Data
############################################################
wanted_clusters = 3
wanted_cluster_nums = np.random.choice(
    ind_frame['cat_ind_fin'].unique(),
    wanted_clusters,
)
wanted_cluster_inds = ind_frame.index[ind_frame.cat_ind_fin
                                      .isin(wanted_cluster_nums)].values

wanted_data = fin_data[wanted_cluster_inds]

# Plot all clusters individually
cluster_plot_inds = [ind_frame.index[ind_frame.cat_ind_fin.isin(
                    [x])].values for x in wanted_cluster_nums]

cluster_dat_list = [fin_data[i] for i in cluster_plot_inds]

max_waves = 1000
fig, ax = plt.subplots(wanted_clusters, 1,
                       sharex=True, sharey=True)
for this_dat, this_ax in zip(cluster_dat_list, ax):
    this_ax.plot(this_dat[:max_waves].T, color='k', alpha=0.1)
plt.show()

############################################################
# Transform Data
############################################################
fe_pipeline = load(os.path.join(path_vars['feature_pipeline_path']))

transformed_data = fe_pipeline.transform(wanted_data)

# Rescale Data (since we don't have to worry about a single model)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(transformed_data)

# Reduce dims to plot in 2D
pca_object = PCA(n_components=2)
pca_data = pca_object.fit_transform(scaled_data)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(scaled_data, aspect='auto', interpolation='nearest')
ax[1].imshow(pca_data, aspect='auto', interpolation='nearest')
ax[2].scatter(*pca_data.T, s=2, alpha=0.5)
plt.show()

############################################################
# Cluster Data
############################################################

estimator = \
    BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=100000,
        n_components=8,
        reg_covar=0,
        init_params="random",
        max_iter=1500,
        mean_precision_prior=0.8,
        random_state=random_state,
    )

estimator.fit(scaled_data)
pred_prob = estimator.predict_proba(scaled_data)
pred_labels = estimator.predict(scaled_data)
# Fraction per cluster
pred_frac = np.bincount(pred_labels) / len(pred_labels)
threshold = 0.02

# Merge clusters below threshold
unwanted_clusters = np.where(pred_frac < threshold)[0]
wanted_clusters = np.where(pred_frac >= threshold)[0]
unwanted_inds = np.where(np.isin(pred_labels, unwanted_clusters))[0]
unwanted_probs = pred_prob[unwanted_inds][:, wanted_clusters]
new_labels = wanted_clusters[np.argmax(unwanted_probs, axis=1)]
fin_pred_labels = pred_labels.copy()
fin_pred_labels[unwanted_inds] = new_labels

# Plot clustered data
cmap = plt.get_cmap('tab10')
fig, ax = plt.subplots(1, 5)
sort_order = np.argsort(fin_pred_labels)
ax[0].imshow(scaled_data[sort_order], aspect='auto', interpolation='nearest')
ax[1].imshow(pca_data[sort_order], aspect='auto', interpolation='nearest')
# scatter plot with colors and legend
for this_clust in np.unique(fin_pred_labels):
    this_inds = np.where(fin_pred_labels == this_clust)[0]
    ax[2].scatter(*pca_data[this_inds].T,
                  color=cmap(this_clust),
                  alpha=0.5,)
ax[2].legend(np.unique(fin_pred_labels))
ax[3].bar(np.arange(len(pred_frac)), pred_frac)
ax[3].axhline(threshold, color='r', linestyle='--')
ax[4].imshow(pred_prob[sort_order], aspect='auto', interpolation='nearest')

# Plot all clusters individually
cluster_plot_inds = [np.where(fin_pred_labels == x)[0]
                     for x in np.unique(fin_pred_labels)]
cluster_dat_list = [wanted_data[i] for i in cluster_plot_inds]
fig, ax = plt.subplots(len(cluster_plot_inds), 1,
                       sharex=True, sharey=True)
for this_dat, this_ax in zip(cluster_dat_list, ax):
    this_ax.plot(this_dat[:max_waves].T, color='k', alpha=0.1)
plt.show()

# Overlay waveforms for clusters 1 and -1
#fig, ax = plt.subplots(1, 1)
#ax.plot(wanted_data[cluster_plot_inds[1]][:max_waves].T, color='k', alpha=0.01)
#ax.plot(wanted_data[cluster_plot_inds[-1]][:max_waves].T, color='r', alpha=0.01)
# plt.show()

# Detect clusters with mixing
# If clusters are similar, their predicted probabilities
# will be correlated
# If clusters are different, their predicted probabilities
# will be negatively correlated (being in one cluster means
# you are not in the other)
cluster_probs = pred_prob[:, wanted_clusters]
cluster_corr = np.corrcoef(cluster_probs.T)
# Change diagonal to Nan
#np.fill_diagonal(cluster_corr, np.nan)

## Plot correlation matrix
#fig, ax = plt.subplots(1, 2)
#im = ax[0].matshow(cluster_corr, aspect='auto', interpolation='nearest')
#plt.colorbar(im, ax=ax[0])
#ax[1].matshow(cluster_corr > 0, aspect='auto', interpolation='nearest')
#plt.show()

## Cluster the cluster_corr using KMeans
#kmeans = KMeans(n_clusters=3, random_state=random_state).fit(cluster_corr)
#kmeans_labels = kmeans.labels_
#
## Plot cluster_corr sorted by kmeans_labels
#sort_inds = np.argsort(kmeans_labels)
#sorted_cluster_corr = cluster_corr[sort_inds][:, sort_inds]
#fig, ax = plt.subplots(1, 1)
#im = ax.matshow(sorted_cluster_corr,
#                aspect='auto', interpolation='nearest')
#plt.colorbar(im, ax=ax)
#plt.show()

## Cluster the cluster_corr using AgglomerativeClustering
# Perform agglomerative clustering
Z = shc.linkage(cluster_corr, 'ward')

## Plot the dendrograms
#fig,ax = plt.subplots(2,1)
#plt.sca(ax[0])
#shc.dendrogram(Z)
#
## Reorder the rows of the matrix based on the clustering
#sorted_idx = shc.dendrogram(Z, no_plot=True)['leaves']
#sorted_matrix = cluster_corr[sorted_idx, :][:, sorted_idx]
#
## Plot the sorted matrix
#ax[1].matshow(sorted_matrix, cmap='gray', aspect='auto')
#plt.show()

# Given linkage matrix, plot dendogram and merged clusters
# at every level

# Map cluster labels to cluster indices
cluster_map = dict(zip(np.unique(fin_pred_labels),
                       np.arange(len(np.unique(fin_pred_labels)))))
z_pred_labels = np.array([cluster_map[x] for x in fin_pred_labels])
cluster_nums = np.unique(z_pred_labels)
orig_count = len(cluster_nums)

cluster_inds_dict = {x: np.where(z_pred_labels == x)[0]
                     for x in cluster_nums}
new_inds = []
for this_row in Z:
    clust1, clust2 = this_row[:2]
    new_clust = np.max(list(cluster_inds_dict.keys())) + 1
    cluster_inds_dict[new_clust] = np.concatenate((cluster_inds_dict[clust1],
                                                   cluster_inds_dict[clust2]))
    new_inds.append(new_clust)

new_Z = np.concatenate((Z, np.array(new_inds)[:, None]), axis=1)

# Create dendogram with cluster data
# Start with original clusters
max_distance = new_Z[:,2].max()

def enumerate_levels(new_Z):
    """
    Assign all nodes to levels in the dendrogram
    """
    # Level array, col1 = node, col2 = level
    levels = np.zeros((int(new_Z[-1,-1]) + 1, 2))
    levels[:,0] = np.arange(len(levels))
    for this_row in new_Z:
        leaves = np.vectorize(np.int)(this_row[:2])
        join = int(this_row[-1])
        #for this_leaf in leaves:
        #    if levels[int(this_leaf), 1] == 0:
        #        pass
        levels[join:,1] = int(np.max(levels[leaves, 1])) + 1
    return np.vectorize(np.int)(levels)

def find_x_pos(new_Z):
    """
    Find the x position of each node in the dendrogram
    """
    leaves = shc.dendrogram(new_Z[:,:-1], no_plot=True)['leaves']
    leave_x_pos = np.arange(len(leaves))
    # x_pos array  col1 = node, col2 = x_pos
    x_pos = np.zeros((int(new_Z[-1,-1]) + 1, 2))
    x_pos[:,0] = np.arange(len(x_pos))
    #x_pos[:,0] = np.vectorize(np.int)(x_pos[:,0])
    x_pos[leaves,1] = leave_x_pos
    for this_row in new_Z:
        leaves = np.vectorize(np.int)(this_row[:2])
        join = int(this_row[-1])
        x_pos[join:,1] = np.mean(x_pos[leaves,1]) 
    return x_pos

new_Z_levels = enumerate_levels(new_Z)[:,1]
new_Z_x_pos = find_x_pos(new_Z)
int_x_pos = np.vectorize(np.int)(new_Z_x_pos[:,1])

from matplotlib.patches import ConnectionPatch

# Plot waveforms along dendogram using levels and x_pos
fig,ax = plt.subplots(np.max(new_Z_levels) + 1, 
                      np.max(int_x_pos) + 1,
                      sharey=True, sharex=True)
for this_clust in list(cluster_inds_dict.keys()):
    this_inds = cluster_inds_dict[this_clust]
    this_level = new_Z_levels[this_clust]
    this_x_pos = int_x_pos[this_clust]
    ax[this_level, this_x_pos].plot(
            wanted_data[this_inds].T[:max_waves], color='k', alpha=0.01)
    ax[this_level, this_x_pos].set_title(this_clust)
    # Remove ticks and spines
for this_ax in ax.flatten():
    this_ax.tick_params(
            axis='both', which='both', bottom=False, top=False,
            labelbottom=False, right=False, left=False, labelleft=False)
    this_ax.spines['right'].set_visible(False)
    this_ax.spines['top'].set_visible(False)
    this_ax.spines['bottom'].set_visible(False)
    this_ax.spines['left'].set_visible(False)
    this_ax.patch.set_alpha(0.0)
# Connect each leaf to its parent
for this_row in new_Z:
    leaves = np.vectorize(np.int)(this_row[:2])
    join = int(this_row[-1])
    for this_leaf in leaves:
        con = ConnectionPatch(xyA=(int_x_pos[this_leaf], new_Z_levels[this_leaf]),
                              xyB=(int_x_pos[join], new_Z_levels[join]),
                              coordsA="data", coordsB="data",
                              axesA=ax[new_Z_levels[this_leaf], int_x_pos[this_leaf]],
                              axesB=ax[new_Z_levels[join], int_x_pos[join]],
                              color="black", alpha=0.5,
                              zorder = 10)
        ax[new_Z_levels[this_leaf], int_x_pos[this_leaf]].add_artist(con)
plt.show()

## Cluster the cluster_corr using AgglomerativeClustering
#agg = AgglomerativeClustering(n_clusters=3).fit(cluster_corr)
#agg_labels = agg.labels_
#
#sort_inds = np.argsort(agg_labels)
#sorted_cluster_corr = cluster_corr[sort_inds][:, sort_inds]
#
## Plot cluster_corr with dendrogram
#fig, ax = plt.subplots(2, 1)
#im = ax[1].matshow(sorted_cluster_corr, 
#                   aspect='auto', interpolation='nearest')
#plt.colorbar(im, ax=ax)
#dend = shc.dendrogram(shc.linkage(cluster_corr, method='ward'),
#                      ax = ax[0],)
#plt.show()
                    

# Get clusters with non-negative correlations
#mixing_cluster_inds = np.where(np.sum(cluster_corr > 0, axis=1) > 0)[0]
#mixing_clusters = wanted_clusters[mixing_cluster_inds]
#print(mixing_clusters)
#
## Another way would be to simply look at what other clusters 
## are predicted for each cluster
#cluster_inds = [np.where(pred_labels == x)[0]
#                for x in np.unique(pred_labels)]
#cluster_probs = np.stack([pred_prob[i].sum(axis=0) \
#        for i in cluster_inds])
## Normalize by the cluster itself
#cluster_probs = cluster_probs / cluster_probs.sum(axis=1)[:, None]
## Change diagonal to Nan
##np.fill_diagonal(cluster_probs, np.nan)
#
## Plot cluster_probs and cluster_probs > threshold
#fig, ax = plt.subplots(1, 2)
#im = ax[0].matshow(cluster_probs, aspect='auto', interpolation='nearest')
#plt.colorbar(im, ax=ax[0])
#ax[1].matshow(cluster_probs > threshold, aspect='auto', interpolation='nearest')
#plt.show()
#
## Cluster the cluster_probs using KMeans
#kmeans = KMeans(n_clusters=3, random_state=random_state).fit(cluster_probs)
#kmeans_labels = kmeans.labels_
#
## Plot clustered cluster_probs
#fig, ax = plt.subplots(1, 2)
#im = ax[0].matshow(cluster_probs, aspect='auto', interpolation='nearest')
#plt.colorbar(im, ax=ax[0])
#
