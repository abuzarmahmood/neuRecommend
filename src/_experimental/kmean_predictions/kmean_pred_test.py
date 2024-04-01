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
from time import time

import sys
sys.path.append('/media/bigdata/projects/neuRecommend/src/create_pipeline')
from feature_engineering_pipeline import *

base_dir = '/media/bigdata/projects/neuRecommend/src/_experimental/kmean_predictions'
plot_dir = f'{base_dir}/plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
artifact_dir = f'{base_dir}/artifacts'
if not os.path.exists(artifact_dir):
    os.mkdir(artifact_dir)

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

clf_path = f'{model_dir}/xgboost_classifier.dump'
clf = load(clf_path)

############################################################
# Kmeans Test
############################################################
# 1) Take subsamples from full dataset
# 2) Cluster into 'k' clusters
# 3) Perform prediction
# 4) Compare mean prediction to actual prediction

n_subsets = 100
n_samples = 10000
# n_samples_per_cluster = np.array([100, 1000])
# n_clusters = n_samples // n_samples_per_cluster 
n_clusters = np.array([10, 100, 250])

recall_list = []
precision_list = []
for this_n_clusters in n_clusters:
    for i in trange(n_subsets): 
        start = time()
        this_subset_inds = np.random.choice(all_data.shape[0], n_samples, replace=False)
        this_subset = all_data[this_subset_inds]
        this_labels = all_labels[this_subset_inds]

        full_pred_proba = pred_pipeline.predict_proba(this_subset)[:, 1]

        # Kmeans
        # Mean values given to classifier need to be in feature-space,
        # not waveform-space, as that is likely to be much less stable
        # Also, median is likely to be more stable than mean
        kmeans = KMeans(n_clusters=this_n_clusters)
        transformed_subset = feature_pipeline.transform(this_subset)
        kmeans.fit(transformed_subset)
        cluster_labels = kmeans.labels_
        processed_centroids = kmeans.cluster_centers_

        cluster_pred_proba = clf.predict_proba(processed_centroids)[:, 1]
        cluster_pred_proba = cluster_pred_proba[cluster_labels]

        # Get prediction
        cluster_pred = cluster_pred_proba > clf_threshold
        full_pred = full_pred_proba > clf_threshold

        # Compare recall between full and cluster
        full_recall = recall_score(this_labels, full_pred)
        cluster_recall = recall_score(this_labels, cluster_pred)

        # Compare precision between full and cluster
        full_precision = precision_score(this_labels, full_pred)
        cluster_precision = precision_score(this_labels, cluster_pred)

        end = time()
        elapsed = end - start

        recall_dict = {
            'cluster_recall': cluster_recall,
            'full_recall': full_recall,
            'n_clusters': this_n_clusters,
            'subset': i,
            'elapsed': elapsed,
        }
        recall_list.append(recall_dict)

        precision_dict = {
            'cluster_precision': cluster_precision,
            'full_precision': full_precision,
            'n_clusters': this_n_clusters,
            'subset': i,
            'elapsed': elapsed,
        }
        precision_list.append(precision_dict)

recall_frame = pd.DataFrame(recall_list)
precision_frame = pd.DataFrame(precision_list)

recall_frame.to_csv(f'{artifact_dir}/recall_frame.csv')
precision_frame.to_csv(f'{artifact_dir}/precision_frame.csv')

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

# Plot histograms of recall and precision
recall_long = recall_frame.melt(
    id_vars=['n_clusters', 'subset', 'elapsed'],
    value_vars=['full_recall', 'cluster_recall'],
    var_name='recall_type',
    value_name='recall_value',
    )
precision_long = precision_frame.melt(
    id_vars=['n_clusters', 'subset', 'elapsed'],
    value_vars=['full_precision', 'cluster_precision'],
    var_name='precision_type',
    value_name='precision_value',
    )

g = sns.displot(
    data=recall_long,
    x='recall_value',
    hue='n_clusters',
    kind='kde',
    row='recall_type',
    )
g.fig.savefig(f'{plot_dir}/recall_comparison_hist_og_thresh.png',
              bbox_inches='tight')
plt.close(g.fig)

g = sns.displot(
    data=precision_long,
    x='precision_value',
    hue='n_clusters',
    kind='kde',
    row='precision_type',
    )
g.fig.savefig(f'{plot_dir}/precision_comparison_hist_og_thresh.png',
              bbox_inches='tight')
plt.close(g.fig)

# Plot elapsed time as histogram
g = sns.displot(
    data=recall_frame,
    x='elapsed',
    hue='n_clusters',
    kind='kde',
    )
g.axes[0].set_xlabels('Elapsed Time (s)')
g.fig.savefig(f'{plot_dir}/elapsed_time_hist_og_thresh.png',
              bbox_inches='tight')
plt.close(g.fig)
