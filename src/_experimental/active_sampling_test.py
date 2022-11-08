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

import sys
sys.path.append('/media/bigdata/projects/neuRecommend/src/create_pipeline')
from feature_engineering_pipeline import *

file_path = '/media/bigdata/projects/neuRecommend/data/final/final_dataset.h5'
h5 = tables.open_file(file_path,'r')

pos_children = h5.get_node('/sorted/pos')._f_iter_nodes()
pos_dat = [x[:] for x in pos_children]
pos_dat = np.concatenate(pos_dat)

neg_children = h5.get_node('/sorted/neg')._f_iter_nodes()
neg_dat = [x[:] for x in neg_children]
neg_dat = np.concatenate(neg_dat)

feature_pipeline_path = '/media/bigdata/projects/neuRecommend/model/feature_engineering_pipeline.dump'
feature_pipeline = load(feature_pipeline_path)

X_raw = np.concatenate([neg_dat,pos_dat])
X = feature_pipeline.transform(X_raw)
y = np.concatenate([np.zeros(len(neg_dat)), np.ones(len(pos_dat))])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

########################################
optim_params_path = '/media/bigdata/projects/neuRecommend/model/optim_params.json'
with open(optim_params_path, 'r') as outfile:
    best_params = json.load(outfile)

############################################################
# Full set
clf = xgb.XGBClassifier(**best_params)
clf.fit(X_train, y_train)
full_set_accuracy = accuracy_score(
        clf.predict(X_test),
        y_test
        )

############################################################
# Random subset 
random_subset_size = int(10e4)
random_inds = np.random.choice(len(X_train), random_subset_size)
random_X, random_y = X_train[random_inds], y_train[random_inds]
clf = xgb.XGBClassifier(**best_params)
clf.fit(random_X, random_y)
random_subset_accuracy = accuracy_score(
        clf.predict(X_test),
        y_test
        )
random_subset_proba = clf.predict_proba(random_X)[:,1]

############################################################
# Activate Learning 
# Smallest margin strategy
# 1) Start with 1000 samples
# 2) Predict on 1e4 samples, and select 1000 most ambiguous ones
# 3) Repeat 100 times

sample_size = int(1e4)
repeats = int(np.ceil(random_subset_size / sample_size))
random_sample_size = int(1e5)

inds = np.random.choice(len(X_train), sample_size)
clf = xgb.XGBClassifier(**best_params)
clf.fit(X_train[inds], y_train[inds])
this_accuracy = accuracy_score(
        clf.predict(X_test),
        y_test
        )

X_step_sample = X_train[inds]
y_step_sample = y_train[inds] 

step_al_dataset_size = [X_step_sample.shape[0]]
step_al_accuracy = [this_accuracy]

for i in trange(repeats-1):
    inds = np.random.choice(len(X_train), random_sample_size)
    this_X, this_y = X_train[inds], y_train[inds]
    preds = clf.predict_proba(this_X)
    wanted_inds = np.argsort(np.abs(np.diff(preds,axis=-1).flatten()))[:sample_size]

    X_step_sample = np.concatenate([X_step_sample, this_X[wanted_inds]])
    y_step_sample = np.concatenate([y_step_sample, this_y[wanted_inds]])

    clf = xgb.XGBClassifier(**best_params)
    clf.fit(X_step_sample, y_step_sample)
    this_accuracy = accuracy_score(
            clf.predict(X_test),
            y_test
            )
    step_al_accuracy.append(this_accuracy)
    step_al_dataset_size.append(X_step_sample.shape[0])

step_al_proba = clf.predict_proba(X_step_sample)[:,1]

############################################################
# Activate Learning 
# Smallest margin strategy
# 1) Start with 1000 samples
# 2) Predict on 1e4 samples, select samples with probability equal to how
#       1-margin
# 3) Repeat 100 times
# **Hoping this serves as a "temperature" parameter, such that we're
# more selective with accepting samples as we progress

sample_size = int(1000)
repeats = int(np.ceil(random_subset_size / sample_size))
random_sample_size = int(1e4)

inds = np.random.choice(len(X_train), sample_size)
clf = xgb.XGBClassifier(**best_params)
clf.fit(X_train[inds], y_train[inds])
this_accuracy = accuracy_score(
        clf.predict(X_test),
        y_test
        )

X_rand_sample = X_train[inds]
y_rand_sample = y_train[inds] 

rand_al_dataset_size = [X_rand_sample.shape[0]]
rand_al_accuracy = [this_accuracy]

for i in trange(repeats-1):
    inds = np.random.choice(len(X_train), random_sample_size)
    this_X, this_y = X_train[inds], y_train[inds]
    preds = clf.predict_proba(this_X)
    margin = np.abs(np.diff(preds,axis=-1).flatten())
    # The smaller the margin, the larger the probability we'll take the sample
    probs = 1-margin
    wanted_inds = np.where(np.random.random(len(probs)) < probs)[0] 

    X_rand_sample = np.concatenate([X_rand_sample, this_X[wanted_inds]])
    y_rand_sample = np.concatenate([y_rand_sample, this_y[wanted_inds]])

    clf = xgb.XGBClassifier(**best_params)
    clf.fit(X_rand_sample, y_rand_sample)
    this_accuracy = accuracy_score(
            clf.predict(X_test),
            y_test
            )
    rand_al_accuracy.append(this_accuracy)
    rand_al_dataset_size.append(X_rand_sample.shape[0])

############################################################
plot_dir = '/media/bigdata/projects/neuRecommend/src/_experimental'

plt.plot(step_al_dataset_size, step_al_accuracy, 
        '-x', label = 'Step AL')
plt.axhline(full_set_accuracy, 
        color = 'red', linestyle = '--', linewidth = 2,
        label = f'Full set : {len(X_train)/step_al_dataset_size[-1]}x')
plt.axhline(random_subset_accuracy, 
        color = 'orange', linestyle = '--', linewidth = 2,
        label = f'Random set : {random_subset_size/step_al_dataset_size[-1]}x')
plt.plot(rand_al_dataset_size, rand_al_accuracy, 
        '-x', label = 'Random AL')
plt.xlabel('Sample size')
plt.ylabel("Test Accuracy")
plt.legend()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'al_accuracy_comparison'))
plt.close(fig)
#plt.show()

feature_labels = [f'ft{x}' for x in range(X_train.shape[1])]
train_frame = pd.DataFrame(
        X_train, 
        columns = feature_labels
        )
random_train_frame = pd.DataFrame(
        random_X, 
        columns = feature_labels
        )
random_train_frame['pred'] = random_subset_proba 
al_train_frame = pd.DataFrame(
        X_step_sample, 
        columns = feature_labels
        )
al_train_frame['pred'] = step_al_proba 

g=sns.pairplot(random_train_frame.iloc[::10,:], hue = 'pred', diag_kind = None);
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'random_subset_dist'))
plt.close(fig)
#g.fig.set_size_inches(5,7)
g=sns.pairplot(al_train_frame.iloc[::10,:], hue = 'pred', diag_kind = None);
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'al_subset_dist'))
plt.close(fig)
#g.fig.set_size_inches(5,7)
#plt.show()
