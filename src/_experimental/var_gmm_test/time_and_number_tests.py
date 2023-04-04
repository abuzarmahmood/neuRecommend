from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import os
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
pipeline_dir = '/media/bigdata/projects/neuRecommend/src/create_pipeline'
sys.path.append(pipeline_dir)
from return_data import return_data
from feature_engineering_pipeline import *

params_dir = os.path.join(pipeline_dir, 'params')
with open(os.path.join(params_dir, 'path_vars.json'), 'r') as path_file:
    path_vars = json.load(path_file)

random_state = 3
rng = np.random.RandomState(random_state)

plot_save_dir = \
    '/media/bigdata/projects/neuRecommend/src/_experimental/var_gmm_test/plots'

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

fe_pipeline = load(os.path.join(path_vars['feature_pipeline_path']))

############################################################
# Select Data
############################################################
b_gmm_run_times = []
gmm_run_times = []
b_gmm_cluster_counts = []
run_cluster_counts = []

repeats = 100
wanted_cluster_list_unique = [1,2,3,4]
wanted_cluster_list = wanted_cluster_list_unique * repeats

for wanted_clusters in tqdm(wanted_cluster_list):
    #wanted_clusters = 3
    wanted_cluster_nums = np.random.choice(
        ind_frame['cat_ind_fin'].unique(),
        wanted_clusters,
    )
    wanted_cluster_inds = ind_frame.index[ind_frame.cat_ind_fin
                                          .isin(wanted_cluster_nums)].values

    wanted_data = fin_data[wanted_cluster_inds]

    ############################################################
    # Transform Data
    ############################################################
    transformed_data = fe_pipeline.transform(wanted_data)

    # Rescale Data (since we don't have to worry about a single model)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(transformed_data)

    ############################################################
    # Cluster Data
    ############################################################

    try:
        b_gmm_start = time()
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
        pred_labels = estimator.predict(scaled_data)
        b_gmm_end = time()
        b_gmm_run_time = b_gmm_end - b_gmm_start

        gmm_start = time()
        estimator = \
            GaussianMixture(
                n_components=8,
                reg_covar=0,
                init_params="random",
                max_iter=1500,
                random_state=random_state,
            )
        estimator.fit(scaled_data)
        pred_labels = estimator.predict(scaled_data)
        gmm_end = time()
        gmm_run_time = gmm_end - gmm_start

        # Fraction per cluster
        pred_frac = np.bincount(pred_labels) / len(pred_labels)
        threshold = 0.02

        detected_cluter_count = np.sum(pred_frac > threshold)

        b_gmm_run_times.append(b_gmm_run_time)
        gmm_run_times.append(gmm_run_time)
        b_gmm_cluster_counts.append(detected_cluter_count)
        run_cluster_counts.append(wanted_clusters)
    except:
        pass

############################################################
# Plot Results
############################################################

# Convert results ot pandas dataframe
results_df = pd.DataFrame(
    dict(
        b_gmm_run_times=b_gmm_run_times,
        gmm_run_times=gmm_run_times,
        b_gmm_cluster_counts=b_gmm_cluster_counts,
        run_cluster_counts=run_cluster_counts,
    )
)
mean_frame = results_df.groupby('run_cluster_counts').mean()


# Plot run times against number of clusters
# and mean run times against number of cluster
# Plot on log scale
# Add jitter to x axis
fig, ax = plt.subplots()
jitter = 0.1
x_jitter = np.random.uniform(-jitter, jitter, len(run_cluster_counts))
ax.scatter(run_cluster_counts + x_jitter, b_gmm_run_times, 
           alpha = 0.5, label='Bayesian GMM')
ax.scatter(run_cluster_counts + x_jitter, gmm_run_times, 
           alpha = 0.5, label='GMM')
ax.plot(mean_frame.index, mean_frame['b_gmm_run_times'],)
ax.plot(mean_frame.index, mean_frame['gmm_run_times'],)
ax.set_yscale('log')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Run Time (s)')
ax.set_title('Run Time vs Number of Clusters')
ax.legend()
fig.savefig(os.path.join(plot_save_dir, 'run_times.png'))

# Plot detected cluster counts against number of clusters
# Make sure axes are the same
# Add jitter to points
jitter = 0.3
fig, ax = plt.subplots()
ax.scatter(run_cluster_counts + np.random.uniform(-jitter, jitter, len(run_cluster_counts)),
           b_gmm_cluster_counts + np.random.uniform(-jitter, jitter, len(b_gmm_cluster_counts)),
           alpha = 0.5,
           )
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Detected Clusters')
ax.set_title('Detected Clusters vs Number of Clusters')
ax.set_ylim([0, 9])
ax.set_xlim([0, 9])
ax.set_aspect('equal')
ax.plot([0, 9], [0, 9], 'k--')
fig.savefig(os.path.join(plot_save_dir, 'detected_clusters.png'))

