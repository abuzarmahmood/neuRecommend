from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import pearsonr

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
b_gmm_cluster_counts = []
run_cluster_counts = []
actual_weight_priors = []

repeats = 10
wanted_cluster_list_unique = [1,2,3,4]
wanted_cluster_list = wanted_cluster_list_unique * repeats

weight_prior_list = [1e-3, 1, 1000, 100000]

product_list = list(product(wanted_cluster_list, weight_prior_list))

#for wanted_clusters in tqdm(wanted_cluster_list):
for this_params in tqdm(product_list):
    #wanted_clusters = 3
    wanted_clusters = this_params[0]
    weight_prior = this_params[1]
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
        estimator = \
            BayesianGaussianMixture(
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=weight_prior,
                n_components=8,
                reg_covar=0,
                init_params="random",
                max_iter=1500,
                mean_precision_prior=0.8,
                random_state=random_state,
            )
        estimator.fit(scaled_data)
        pred_labels = estimator.predict(scaled_data)

        # Fraction per cluster
        pred_frac = np.bincount(pred_labels) / len(pred_labels)
        threshold = 0.02

        detected_cluter_count = np.sum(pred_frac > threshold)

        b_gmm_cluster_counts.append(detected_cluter_count)
        run_cluster_counts.append(wanted_clusters)
        actual_weight_priors.append(weight_prior)
    except:
        pass

############################################################
# Plot Results
############################################################

# Convert results ot pandas dataframe
results_df = pd.DataFrame(
    dict(
        b_gmm_cluster_counts=b_gmm_cluster_counts,
        run_cluster_counts=run_cluster_counts,
        actual_weight_priors=actual_weight_priors,
    )
)

# For each weight prior, calculate the correlation between the
# number of clusters detected and the number of clusters run
# Plot the results
fig, ax = plt.subplots(len(weight_prior_list), 1, figsize=(5, 10))
for i, weight_prior in enumerate(weight_prior_list):
    this_df = results_df[results_df['actual_weight_priors'] == weight_prior]
    corr = pearsonr(this_df['b_gmm_cluster_counts'],
                    this_df['run_cluster_counts'])
    jitter_x = rng.normal(0, 0.3, len(this_df))
    jitter_y = rng.normal(0, 0.3, len(this_df))
    ax[i].scatter(this_df['b_gmm_cluster_counts'] + jitter_x,
                  this_df['run_cluster_counts'] + jitter_y)
    ax[i].set_title(f'Weight Prior: {weight_prior}, Corr: {np.round(corr[0], 2)}')
    ax[i].set_xlabel('Detected Clusters')
    ax[i].set_ylabel('Actual Clusters')
    ax[i].set_xlim([0, 8])
    ax[i].set_ylim([0, 8])
    ax[i].set_aspect('equal')
    ax[i].plot([0, 8], [0, 8], 'k--')
plt.tight_layout()
plt.savefig(os.path.join(plot_save_dir, 'weight_prior_test.png'))
plt.close(fig)
