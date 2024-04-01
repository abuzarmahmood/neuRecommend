import tables
import os
import numpy as np
from joblib import load
import seaborn as sns
import pandas as pd
import pylab as plt
import xgboost as xgb
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, precision_score

import sys
sys.path.append('/media/bigdata/projects/neuRecommend/src/create_pipeline')
from feature_engineering_pipeline import *

plot_dir = '/media/bigdata/projects/neuRecommend/src/_experimental/kmean_predictions/plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

file_path = '/media/bigdata/projects/neuRecommend/data/final/final_dataset.h5'
h5 = tables.open_file(file_path,'r')

pos_children = h5.get_node('/sorted/pos')._f_iter_nodes()
pos_dat = [x[:] for x in pos_children]
pos_dat = np.concatenate(pos_dat)
pos_labels = np.ones(pos_dat.shape[0])

neg_children = h5.get_node('/sorted/neg')._f_iter_nodes()
neg_dat = [x[:] for x in neg_children]
neg_dat = np.concatenate(neg_dat)
neg_labels = np.zeros(neg_dat.shape[0])

all_data = np.concatenate([pos_dat, neg_dat])
all_labels = np.concatenate([pos_labels, neg_labels])

############################################################
# Get model

home_dir = os.environ.get("HOME")
model_dir = f'{home_dir}/Desktop/neuRecommend/model'

pred_pipeline_path = f'{model_dir}/xgboost_full_pipeline.dump'
feature_pipeline_path = f'{model_dir}/feature_engineering_pipeline.dump'

clf_threshold_path = f'{model_dir}/proba_threshold.json'
with open(clf_threshold_path, 'r') as this_file:
    out_dict = json.load(this_file)
clf_threshold = out_dict['threshold']

feature_pipeline = load(feature_pipeline_path)
pred_pipeline = load(pred_pipeline_path)
# clf_prob = pred_pipeline.predict_proba(slices)[:, 1]

############################################################
# Kmeans Test
############################################################
# 1) Take subsamples from full dataset
# 2) Cluster into 'k' clusters
# 3) Perform prediction
# 4) Compare mean prediction to actual prediction

n_subsets = 100
n_samples = 10000
n_samples_per_cluster = np.array([100, 1000])
n_clusters = n_samples // n_samples_per_cluster 

recall_list = []
precision_list = []
for this_n_clusters in n_clusters:
    for i in trange(n_subsets): 
        this_subset_inds = np.random.choice(all_data.shape[0], n_samples, replace=False)
        this_subset = all_data[this_subset_inds]
        this_labels = all_labels[this_subset_inds]

        full_pred_proba = pred_pipeline.predict_proba(this_subset)[:, 1]

        # Kmeans
        kmeans = KMeans(n_clusters=this_n_clusters)
        transformed_subset = feature_pipeline.transform(this_subset)
        kmeans.fit(this_subset)
        cluster_labels = kmeans.labels_
        processed_centroids = kmeans.cluster_centers_

        # Get centroids
        centroids = []
        for clust_ind in range(this_n_clusters):
            this_cluster_inds = np.where(cluster_labels == clust_ind)[0]
            this_cluster_data = this_subset[this_cluster_inds]
            this_cluster_mean = np.mean(this_cluster_data, axis=0)
            centroids.append(this_cluster_mean)
        centroids = np.array(centroids)

        cluster_pred_proba = pred_pipeline.predict_proba(centroids)[:, 1]
        cluster_pred_proba = cluster_pred_proba[cluster_labels]

        # Get prediction
        cluster_pred = cluster_pred_proba > clf_threshold
        full_pred = full_pred_proba > clf_threshold

        # Compare recall between full and cluster
        full_recall = recall_score(this_labels, full_pred)
        cluster_recall = recall_score(this_labels, cluster_pred)

        recall_dict = {
            'cluster_recall': cluster_recall,
            'full_recall': full_recall,
            'n_clusters': this_n_clusters,
            'subset': i,
        }
        recall_list.append(recall_dict)

        # Compare precision between full and cluster
        full_precision = precision_score(this_labels, full_pred)
        cluster_precision = precision_score(this_labels, cluster_pred)

        precision_dict = {
            'cluster_precision': cluster_precision,
            'full_precision': full_precision,
            'n_clusters': this_n_clusters,
            'subset': i,
        }
        precision_list.append(precision_dict)

recall_frame = pd.DataFrame(recall_list)
precision_frame = pd.DataFrame(precision_list)

sns.relplot(
    data=recall_frame,
    x = 'full_recall',
    y = 'cluster_recall',
    hue = 'n_clusters',
    kind = 'scatter',
    )
ax = plt.gca()
ax.set_aspect('equal')
# Rotate the x-axis labels
plt.xticks(rotation=45)
fig = plt.gcf()
fig.savefig(f'{plot_dir}/recall_comparison_og_thresh.png',
            bbox_inches='tight')
plt.close(fig)
# plt.show()

# Separate by n_clusters
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
for i, this_n_clusters in enumerate(np.sort(n_clusters)):
    this_frame = recall_frame[recall_frame['n_clusters'] == this_n_clusters]
    min_val = np.min([this_frame['full_recall'].min(), this_frame['cluster_recall'].min()])
    max_val = np.max([this_frame['full_recall'].max(), this_frame['cluster_recall'].max()])
    ax[i].scatter(this_frame['full_recall'], this_frame['cluster_recall'], label=this_n_clusters)
    ax[i].plot([min_val, max_val], [min_val, max_val], 'k--')
    ax[i].set_aspect('equal')
    ax[i].set_title(f'N Clusters: {this_n_clusters}')
    ax[i].set_xlabel('Full Recall')
    ax[i].set_ylabel('Cluster Recall')
fig.savefig(f'{plot_dir}/recall_comparison_og_thresh_separate.png',
            bbox_inches='tight')
plt.close(fig)

sns.relplot(
    data=precision_frame,
    x = 'full_precision',
    y = 'cluster_precision',
    hue = 'n_clusters',
    kind = 'scatter',
    )
ax = plt.gca()
ax.set_aspect('equal')
# Rotate the x-axis labels
plt.xticks(rotation=45)
fig = plt.gcf()
fig.savefig(f'{plot_dir}/precision_comparison_og_thresh.png',
            bbox_inches='tight')
plt.close(fig)

# Separate by n_clusters
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
for i, this_n_clusters in enumerate(np.sort(n_clusters)):
    this_frame = precision_frame[precision_frame['n_clusters'] == this_n_clusters]
    min_val = np.min([this_frame['full_precision'].min(), this_frame['cluster_precision'].min()])
    max_val = np.max([this_frame['full_precision'].max(), this_frame['cluster_precision'].max()])
    ax[i].scatter(this_frame['full_precision'], this_frame['cluster_precision'], label=this_n_clusters)
    ax[i].plot([min_val, max_val], [min_val, max_val], 'k--')
    ax[i].set_aspect('equal')
    ax[i].set_title(f'N Clusters: {this_n_clusters}')
    ax[i].set_xlabel('Full Precision')
    ax[i].set_ylabel('Cluster Precision')
fig.savefig(f'{plot_dir}/precision_comparison_og_thresh_separate.png',
            bbox_inches='tight')
plt.close(fig)
