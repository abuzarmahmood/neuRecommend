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
from sklearn.metrics import recall_score, precision_score, accuracy_score
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

import sys
sys.path.append('/media/bigdata/projects/neuRecommend/src/create_pipeline')
from feature_engineering_pipeline import *

plot_dir = '/media/bigdata/projects/neuRecommend/src/_experimental/spectral_features/plots'
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
# Calculate fft on samples

n_samples = 10000
n_repeats = 100
cv = 5

accuracy_list = []
fraction_list = []
for i in trange(n_repeats):
    # Amplitude normalization of waveforms
    scaler = StandardScaler()
    scaled_all_data = scaler.fit_transform(all_data.T).T

    sample_inds = np.random.choice(all_data.shape[0], n_samples, replace=False)
    this_data = scaled_all_data[sample_inds]
    this_labels = all_labels[sample_inds]
    spike_fraction = np.mean(this_labels)
    fraction_list.append(spike_fraction)

    fft_data = fft(this_data)
    freq_vec = fftfreq(this_data.shape[1])
    re_fft_data = np.real(fft_data)
    im_fft_data = np.imag(fft_data)

    # plt.plot(re_fft_data[0])
    # plt.plot(im_fft_data[0])
    # plt.show()
    # 
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(re_fft_data, aspect='auto')
    # ax[1].imshow(im_fft_data, aspect='auto')
    # plt.show()

    # Concatenate real and imaginary parts and perform PCA
    cat_fft_data = np.concatenate([re_fft_data, im_fft_data], axis=1)
    pca = PCA(n_components=0.9)
    pca.fit(cat_fft_data)
    pca_fft_data = pca.transform(cat_fft_data)

    # Perform classification

    # Scale PCA features
    feature_scaler = StandardScaler()
    scaled_pca_fft_data = feature_scaler.fit_transform(pca_fft_data)

    # Calculate cross-validation score using xgboost
    clf = xgb.XGBClassifier()
    scores = cross_val_score(clf, scaled_pca_fft_data, this_labels, cv=cv, scoring='accuracy')
    accuracy_list.append(scores)

fin_fraction_list = [[x]*cv for x in fraction_list]
fin_fraction_list = [x for sublist in fin_fraction_list for x in sublist]

fin_accuracy_list = [x for sublist in accuracy_list for x in sublist]

# Plot accuracy vs spike fraction
fig, ax = plt.subplots(1, 2, figsize=(10, 5),
                       sharey=True)
ax[0].scatter(fin_fraction_list, fin_accuracy_list)
ax[0].set_xlabel('Fraction of spikes')
ax[0].set_ylabel('Cross-validated Accuracy')
ax[1].hist(fin_accuracy_list, bins=20, orientation='horizontal')
plt.savefig(f'{plot_dir}/fft_accuracy_vs_fraction.png')
plt.close()

##############################
# Generate clustermap using sns with row colors

cmap = sns.color_palette("husl", 2)
label_colors = np.array([cmap[int(x)] for x in this_labels])
sns.clustermap(pca_fft_data, row_colors=label_colors, figsize=(10, 10))
# plt.show()
plt.savefig(f'{plot_dir}/fft_clustermap.png')
plt.close()
