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
from sklearn.decomposition import PCA
from scipy.spatial import distance

import sys
sys.path.append('/media/bigdata/projects/neuRecommend/src/create_pipeline')
from feature_engineering_pipeline import *

plot_dir = '/media/bigdata/projects/neuRecommend/src/_experimental/best_classification_times/plots'
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
mean_abs_coef = np.abs(coef_list).mean(axis=0)

# Scale mean_abs_coef to be between 0 and 1
mean_abs_coef = (mean_abs_coef - mean_abs_coef.min()) / (mean_abs_coef.max() - mean_abs_coef.min())

# Plot
fig,ax = plt.subplots(3,1, sharex=True)
ax[1].imshow(np.abs(coef_list), aspect='auto', cmap='viridis')
ax[1].colorbar()
ax[0].plot(mean_abs_coef, '-x')
fig.suptitle('Logistic regression coefficients')
ax[0].set_ylabel('Mean ABS coefficient')
ax[1].set_ylabel('Run #')
ax[1].set_xlabel('Timepoint')
ax[2].imshow(coef_list, aspect='auto', cmap='viridis')
plt.show()
# fig.savefig(os.path.join(plot_dir, 'logistic_regression_coefficients.png'))
# plt.close(fig)

# PCA of coefficients
pca_obj = PCA(n_components=6)
pca_obj.fit(coef_list.T)
pca_coef = pca_obj.transform(coef_list.T)
var_exp = pca_obj.explained_variance_ratio_

# plt.plot(pca_coef, linewidth=5, alpha=0.7)
fig, ax = plt.subplots(1,2, figsize=(15,5))
for i, this_dat in enumerate(pca_coef.T):
    ax[0].plot(this_dat, alpha=0.7, label=f'PC {i}', linewidth=5)
ax[0].legend()
# ax[0].imshow(pca_coef.T, aspect='auto', cmap='viridis')
ax[0].set_title('PCA of logistic regression coefficient')
ax[0].set_xlabel('Timepoint')
ax[0].set_ylabel('PC value')
ax[1].plot(np.cumsum(var_exp), '-x')
ax[1].set_title('Explained variance')
ax[1].set_xlabel('PC #')
ax[1].set_ylabel('Cumulative Explained variance')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'pca_of_logistic_regression_coefficients.png'),
            bbox_inches='tight', dpi=300)
plt.close(fig)

# plt.plot(coef_list.T, alpha = 0.01, color='k')
# plt.show()


# Test projection using weighted and unweighted coefficients
# fig, ax = plt.subplots(2,1, sharex=True)
# cmap = plt.cm.get_cmap('viridis')
# for this_x, this_y in zip(X, y):
#         ax[0].plot(this_x, color=cmap(this_y),
#                  alpha = 0.01)
# ax[1].plot(np.abs(lr.coef_.flatten()))
# fig.suptitle('Logistic regression cofficients \n'
#              f'Accuracy: {lr.score(X, y)}')
# plt.show()

fig,ax = plt.subplots(2,1, sharex=True)
ax[0].plot(pca_obj.components_.T)
ax[1].plot(mean_abs_coef)
plt.show()

cos_dists = [distance.cosine(x, mean_abs_coef) for x in pca_obj.components_]

unweighted_acc = []
weighted_acc = []
joint_acc = []
weighted_pca_prob = []
unweighted_pca_prob = []
for i in trange(n_runs):
    this_unit_inds = unit_inds[i]
    this_dat = [norm_dat[x] for x in this_unit_inds]
    this_labels = [np.ones(len(x)) * i for i, x in enumerate(this_dat)]
    X = np.concatenate(this_dat)
    y = np.concatenate(this_labels)

    # Unweighted
    pca_obj = PCA(n_components=6)
    pca_obj.fit(X)
    pca_proj = pca_obj.transform(X)
    lr = LR()
    lr.fit(pca_proj, y)
    unweighted_acc.append(lr.score(pca_proj, y))

    # weighted
    # weighted_x = X * mean_abs_coef 
    # weighted_pca_obj = PCA(n_components=3)
    # weighted_pca_obj.fit(weighted_x)
    # weighted_pca_proj = weighted_pca_obj.transform(weighted_x)
    weighted_pca_proj = X @ pca_coef
    lr = LR()
    lr.fit(weighted_pca_proj, y)
    weighted_acc.append(lr.score(weighted_pca_proj, y))

    # # Joint
    joint_pca_proj = np.concatenate(
            [pca_proj[:,:3], weighted_pca_proj[:,:3]], axis=1) 
    lr = LR()
    lr.fit(joint_pca_proj, y)
    joint_acc.append(lr.score(joint_pca_proj, y))


unweighted_acc = np.array(unweighted_acc)
weighted_acc = np.array(weighted_acc)
joint_acc = np.array(joint_acc)
all_acc = np.stack([unweighted_acc, weighted_acc, joint_acc])

##############################
acc_diff = weighted_acc - unweighted_acc
diff_sort_inds = np.argsort(acc_diff)[::-1]
n_top = 12
top_diff_inds = diff_sort_inds[:n_top]
top_diff_pairs = [unit_inds[x] for x in top_diff_inds] 

# Plot top diff pairs
cmap = plt.cm.get_cmap('tab10')
n_rows = int(np.sqrt(n_top))
n_cols = int(np.ceil(n_top / n_rows))
fig, ax = plt.subplots(n_rows, n_cols, figsize=(15,10),
                       sharex=True, sharey=True)
for i, this_pair in enumerate(top_diff_pairs):
    for j, this_unit in enumerate(this_pair):
        mean_dat = norm_dat[this_unit].mean(axis=0)
        std_dat = norm_dat[this_unit].std(axis=0)
        # ax.flatten()[i].plot(norm_dat[this_unit].T, alpha=0.05, c = cmap(j))
        ax.flatten()[i].plot(mean_dat, c = cmap(j))
        ax.flatten()[i].fill_between(np.arange(len(mean_dat)),
                                     y1=mean_dat - std_dat,
                                     y2=mean_dat + std_dat,
                                     alpha=0.5, color=cmap(j))
        ax.flatten()[i].set_title(f'Units {", ".join([str(x) for x in this_pair])}' + '\n' +\
                f'Delta acc: {acc_diff[top_diff_inds[i]]:.3f}')
        ax.flatten()[i].axis('off')
fig.suptitle('Top 12 pairs of units with highest weighted - unweighted accuracy')
plt.savefig(os.path.join(plot_dir, 'top_12_pairs_of_units_with_highest_weighted_unweighted_accuracy.png'),
            bbox_inches='tight', dpi=300)
plt.close(fig)
# plt.show()

##############################

print(f'Unweighted accuracy: {unweighted_acc.mean()}')
print(f'Weighted accuracy: {weighted_acc.mean()}')
print(f'Joint accuracy: {joint_acc.mean()}')

plt.hist(all_acc.T, bins=10, label=['Unweighted', 'Weighted', 'Joint'])
plt.legend()
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.title('Logistic regression accuracy...6 components per type')
plt.savefig(os.path.join(plot_dir, 'logistic_regression_accuracy.png'))
plt.close()
# plt.show()

plt.hist(unweighted_acc, bins=50, alpha=0.5, label='Unweighted')
plt.hist(weighted_acc, bins=50, alpha=0.5, label='Weighted')
plt.hist(joint_acc, bins=50, alpha=0.5, label='Joint')
plt.legend()
plt.show()

fig, ax = plt.subplots(2,1, figsize=(5,10))
ax[0].scatter(unweighted_acc, weighted_acc, alpha=0.7, s = 2, color='k')
# Plot x=y
min_val = np.min([unweighted_acc.min(), weighted_acc.min()])
max_val = np.max([unweighted_acc.max(), weighted_acc.max()])
ax[0].plot([min_val, max_val], [min_val, max_val], color='r', linestyle='--')
ax[0].set_xlabel('Unweighted accuracy')
ax[0].set_ylabel('Weighted accuracy')
acc_diff = unweighted_acc - weighted_acc
mean_acc_diff = np.mean(acc_diff)
median_acc_diff = np.median(acc_diff)
ax[1].hist(acc_diff, bins = 50)
ax[1].axvline(0, color='r', linestyle='--')
# Plot arrow for median
# ax[1].annotate('Mean: {:.2f}'.format(mean_acc_diff), 
#                xy=(mean_acc_diff, 0), xytext=(mean_acc_diff, 10),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
ax[1].set_xlabel('Unweighted - Weighted accuracy \n' +\
        ' <- Weighted better | Unweighted better ->')
ax[1].set_ylabel('Count')
ax[0].set_aspect('equal')
ax[1].axis('off')
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'logistic_regression_accuracy_comparison.png'),
            bbox_inches='tight', dpi=300)
plt.close()
#plt.show()

# fig, ax = plt.subplots(1,2,)# sharex=True, sharey=True)
# ax[0].scatter(pca_proj[:,0], pca_proj[:,1], c=y, alpha=0.1)
# ax[1].scatter(weighted_pca_proj[:,0], weighted_pca_proj[:,1], c=y, alpha=0.1)
# plt.show()
