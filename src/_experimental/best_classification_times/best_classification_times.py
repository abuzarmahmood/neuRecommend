"""
For waveforms, find timepoints which are best for classification.

This can be done using the following methods:
         - LDA
         - Neighbourhood Components Analysis (NCA)
         - XGBoost with feature importance

Compare pairs of units and generate distributions of timepoints where
classification is best.

Since knowing this would be most useful for waveforms of the 
same amplitude, normalize the waveforms to the same amplitude.
"""
import tables
import os
import numpy as np
from joblib import load
import seaborn as sns
import pandas as pd
import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from tqdm import tqdm, trange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from collections import Counter
from sklearn.linear_model import LogisticRegression as LR

import sys
sys.path.append('/media/bigdata/projects/neuRecommend/src/create_pipeline')
from feature_engineering_pipeline import *

plot_dir = '/media/bigdata/projects/neuRecommend/src/_experimental/best_classification_times'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)


file_path = '/media/bigdata/projects/neuRecommend/data/final/final_dataset.h5'
h5 = tables.open_file(file_path,'r')

pos_children = h5.get_node('/sorted/pos')._f_iter_nodes()
pos_dat = [x[:] for x in pos_children]
# pos_dat = np.concatenate(pos_dat)

neg_children = h5.get_node('/sorted/neg')._f_iter_nodes()
neg_dat = [x[:] for x in neg_children]
# neg_dat = np.concatenate(neg_dat)

all_dat = pos_dat + neg_dat

# Normalize all waveforms to same amplitude

# center_ind = [[np.argmax(np.abs(y)) for y in x] for x in all_dat]
# center_ind = [x for y in center_ind for x in y]
# center_counter = Counter(center_ind)

center_ind = 30
norm_dat = []
for x in tqdm(all_dat):
        norm_dat.append(x/np.mean(x[:, center_ind]))

# For all combinations of units, find best timepoints for classification
n_runs = 1000
unit_inds = [np.random.choice(len(norm_dat), 2, replace=True) for x in range(n_runs)]

# i = 0
coef_list = []
score_list = []
for i in trange(n_runs):
    this_unit_inds = unit_inds[i]
    this_dat = [norm_dat[x] for x in this_unit_inds]
    this_labels = [np.ones(len(x)) * i for i, x in enumerate(this_dat)]
    X = np.concatenate(this_dat)
    y = np.concatenate(this_labels)

    lr = LR()
    lr.fit(X, y)
    coef_list.append(lr.coef_.flatten())
    score_list.append(lr.score(X, y))

coef_list = np.array(coef_list)
score_list = np.array(score_list)

# Sort by score
sort_inds = np.argsort(score_list)[::-1]
coef_list = coef_list[sort_inds]
score_list = score_list[sort_inds]

# Plot
fig,ax = plt.subplots(2,1, sharex=True)
ax[1].imshow(np.abs(coef_list), aspect='auto', cmap='viridis')
ax[1].colorbar()
ax[0].plot(np.abs(coef_list).mean(axis=0), '-x')
fig.suptitle('Logistic regression coefficients')
ax[0].set_ylabel('Mean ABS coefficient')
ax[1].set_ylabel('Run #')
ax[1].set_xlabel('Timepoint')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'logistic_regression_coefficients.png'))
plt.close(fig)

fig, ax = plt.subplots(2,1, sharex=True)
cmap = plt.cm.get_cmap('viridis')
for this_x, this_y in zip(X, y):
        ax[0].plot(this_x, color=cmap(this_y),
                 alpha = 0.01)
ax[1].plot(np.abs(lr.coef_.flatten()))
fig.suptitle('Logistic regression cofficients \n'
             f'Accuracy: {lr.score(X, y)}')
plt.show()
