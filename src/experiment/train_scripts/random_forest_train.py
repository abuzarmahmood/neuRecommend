"""
https://docs.neptune.ai/getting-started/how-to-add-neptune-to-your-code
"""

import tables
import numpy as np
from tqdm import tqdm
import os
import matplotlib
# matplotlib.use('TKAgg')
import pylab as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA as pca
from time import time
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import shap
import multiprocessing

import neptune.new as neptune
from neptune.new.types import File
from neptune.new.integrations.xgboost import NeptuneCallback

from joblib import dump, load

############################################################
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################
with open('neptune_params.json','r') as path_file:
    neptune_params = json.load(path_file)
run = neptune.init(**neptune_params,
        source_files = ["random_forest_train.py"],
        capture_stdout = False,
        capture_stderr = False,
        capture_hardware_metrics = False)
                                               
with open('path_vars.json','r') as path_file:
    path_vars = json.load(path_file)

h5_path = path_vars['h5_fin_path']
h5_dir = os.path.dirname(h5_path)
model_save_dir = path_vars['model_save_dir']
plot_dir = path_vars['plot_dir']

run["data/train_dataset"].track_files(h5_path)
run["data/train_dataset_metadata"].upload(
        os.path.join(h5_dir, 'fin_data_metadata.csv'))
run["data/train_dataset_summary_stats"].upload(
        os.path.join(h5_dir, 'fin_data_summary_stats.json'))

# Load equal numbers of waveforms for pos,neg, split into train,test
# Since positive samples are >> negative, we will subsample from them
neg_path = '/sorted/neg'
pos_path = '/sorted/pos'

neg_waveforms = []
pos_waveforms = []

with tables.open_file(h5_path,'r') as h5:
    #h5 = tables.open_file(h5_path, 'r')
    for x in h5.iter_nodes(neg_path):
        neg_waveforms.append(x[:])
    for x in h5.iter_nodes(pos_path):
        pos_waveforms.append(x[:])

neg_waveforms = np.concatenate(neg_waveforms, axis=0)
pos_waveforms = np.concatenate(pos_waveforms, axis=0)

neg_label = [0]*neg_waveforms.shape[0]
pos_label = [1]*pos_waveforms.shape[0]
fin_labels = np.concatenate([neg_label, pos_label])

############################################################
# Train classifier
############################################################

def zscore_custom(x):
    return zscore(x, axis=-1)

zscore_transform = FunctionTransformer(zscore_custom)

fin_data = np.concatenate([neg_waveforms, pos_waveforms])
#zscore_fin_data = zscore(fin_data, axis=-1)
zscore_fin_data2 = zscore_transform.transform(fin_data)
pca_obj = pca(n_components=10).fit(zscore_fin_data2[::1000])
print(f'Explained variance : {np.sum(pca_obj.explained_variance_ratio_)}')
pca_data = pca_obj.transform(zscore_fin_data2)

scaler_obj = StandardScaler().fit(pca_data)
X = scaler_obj.transform(pca_data)
y = fin_labels

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, random_state=1)

X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.25, random_state=1)

model_type = 'random_forest'
#param_grid = {
#    'max_depth': [4, 6],
#    'max_features': [3, 6],
#    'n_estimators': [10, 50]
#    }
#rf_model = RandomForestClassifier()
#clf = GridSearchCV(estimator = rf_model, param_grid = param_grid, 
#                      cv = 3, n_jobs = -1, verbose = 2)
#clf.fit(X_train, y_train)
#
## Write out best_params to json
optim_params_path = os.path.join(model_save_dir, f'{model_type}_optim_params.json')
#with open(optim_params_path,'w') as outfile:
#    json.dump(clf.best_params_, outfile, indent = 4)

with open(optim_params_path, 'r') as outfile:
    best_params = json.load(outfile)
run["model/parameters"] = best_params

clf = RandomForestClassifier(**best_params)

train_start = time()
clf.fit(X_train, y_train)
train_end = time()
fit_time = train_end - train_start
run['model/type'] = model_type 
run["model/fit_time"] = np.round(fit_time, 3) 
run["model/train_set_size"] = X_train.shape 

#clf = load(os.path.join(model_save_dir, 'xgb_classifier'))
pred_start = time()
train_score = clf.score(X_train, y_train)
pred_end = time()
pred_time = pred_end-pred_start
run["model/train_set_pred_time"] = np.round(pred_time, 3) 

#test_score = clf.score(X_test, y_test)

#run["evaluation/test_accuracy"] = test_score

############################################################
# Titrate decision boundaries
############################################################
val_proba = clf.predict_proba(X_val)[:, 1]
all_proba = clf.predict_proba(X)[:, 1]

# Priorities in order of importance
#   1) Do not lose any actual neurons (Maximize Recall)
#   2) Discard as many non-neurons as possible
#   3) Make meaningfull recommendations (Maximize precision)
# We can have different thresholds for 1+2, and 3.
# For 1+2, we will SET a threshold for how many true positives
# we want to have, and maximize the number of false positives we can remove
# For 3, we wil want to maximize the AUC for balance the Precision/Recall
# trade-off for recommendations while erring on the side of false positives
# Threshold for true positives == 99%
wanted_thresh = 0.99

thresh_vec = np.linspace(0, 1, 100)
true_mean_val = []
for this_thresh in thresh_vec:
    pred_pos = val_proba >= this_thresh
    pred_pos = pred_pos*1
    pred_pos_mean = np.mean(pred_pos[np.where(y_val)[0]])
    true_mean_val.append(pred_pos_mean)
true_mean_val = np.array(true_mean_val)


false_num, proba = np.histogram(val_proba[np.where(y_val == 0)[0]], bins=100)
cumu_false = np.cumsum(false_num)
scaled_cumu_false = cumu_false/np.max(cumu_false)

true_mean_thresh_inds = true_mean_val >= wanted_thresh
highest_thresh = np.max(thresh_vec[true_mean_thresh_inds])
best_false = scaled_cumu_false[np.argmin(np.abs(proba-highest_thresh))]

fin_val_pred = val_proba > highest_thresh
labels = ['true_neg','false_neg','false_pos','true_pos']
conf_mat = confusion_matrix(y_val, fin_val_pred, normalize = 'true')
conf_dict = dict(zip(labels, np.round(conf_mat.ravel(),4)))
print(conf_dict)

thresh_accuracy = np.mean((val_proba > highest_thresh) == y_val)

run["model/parameters/final_threshold"] = highest_thresh
run["evaluation/final_accuracy"] = thresh_accuracy 
run["evaluation/final_confusion_matrix"] = conf_dict

############################################################
# Plot distribution of predicted probabilities
############################################################
left_str = '1% Spikes' + '\n' + f'{conf_dict["true_neg"]*100:.1f}% Noise'
right_str = f'99% Spikes' + '\n' + f'{100 - (conf_dict["true_neg"]*100):.1f}% Noise'

plt.hist(val_proba[np.where(y_val)[0]], bins=50,
         alpha=0.5, label='True Spike')
# plt.yscale('log')
plt.hist(val_proba[np.where(y_val == 0)[0]],
         alpha=0.5, bins=50, label='True Noise')
plt.xlabel('Spike probability')
plt.ylabel('Frequency (Log Scale)')
plt.axvline(highest_thresh,
            color='red', alpha=0.7, linewidth=2,
            label=f'Threshold for {wanted_thresh} Recall')
plt.text(0.05, 0.6, left_str, transform=plt.gca().transAxes)
plt.text(0.25, 0.6, right_str, transform=plt.gca().transAxes)
plt.title('Distribution of classification probabilities')
plt.legend()
plt.savefig(os.path.join(plot_dir, f'{model_type}_spike_classification_dist.png'),
            dpi=300, bbox_inches='tight')
plt.close()
# plt.show()

############################################################
# Representative plots of spikes and noise at different probs
############################################################
wanted_probs = np.linspace(0, 1, 6)
prob_frame = pd.DataFrame(
    dict(
        label=y,
        prob=all_proba
    )
)

#nrn_inds = []
#noise_inds = []
inds = []
for this_prob in wanted_probs:
    #spike_val = (prob_frame['prob'][prob_frame.label == 1] - this_prob)\
    #    .abs().argsort()[:10].values
    #noise_val = (prob_frame['prob'][prob_frame.label == 0] - this_prob)\
    #    .abs().argsort()[:10].values
    val = (prob_frame['prob'] - this_prob)\
        .abs().argsort()[:10].values
    #nrn_inds.append(spike_val)
    #noise_inds.append(noise_val)
    inds.append(val)

fig, ax = plt.subplots(1, len(wanted_probs), sharey=True,
                       figsize=(13, 2))
for num in range(len(wanted_probs)):
    # ax[0,num].plot(fin_data[nrn_inds[num]])
    # ax[1,num].plot(fin_data[noise_inds[num]])
    this_dat = zscore(fin_data[inds[num]], axis=-1)
    flip_bool = np.vectorize(np.int)(np.sign(this_dat[:, 30]))
    this_dat = np.stack([x*-1 if this_bool == 1 else x
                         for x, this_bool in zip(this_dat, flip_bool)])
    ax[num].plot(this_dat.T,
                 color='k', alpha=0.5)
    ax[num].set_title(f'Prob : {np.round(wanted_probs[num],1)}')
    ax[num].set_xticklabels([])
    ax[num].set_xticks([])
    ax[num].set_yticklabels([])
    ax[num].set_yticks([])
plt.savefig(os.path.join(plot_dir, 
                f'{model_type}_spike_classification_examples.png'),
            dpi=300)
plt.close()
# plt.show()

#Log sample predictions
sample_predictions = []
for this_prob, this_set in zip(wanted_probs, inds):
    for this_ind in this_set:
        this_dat = fin_data[this_ind] 
        this_prob = prob_frame['prob'].iloc[this_ind]
        this_label = prob_frame['label'].iloc[this_ind]
        this_dict = dict(
                dat = this_dat,
                prob = this_prob,
                label = this_label
                )
        sample_predictions.append(this_dict)

sample_frame = pd.DataFrame(sample_predictions)
sample_frame['pred_label'] = (sample_frame['prob'] > highest_thresh)*1

#run["evaluation/predictions"].upload(File.as_html(sample_frame))

#for x in sample_predictions:
#    dat = x['dat']
#    prob = x['prob']
#    label = x['label']
#    pred_label = int(prob >=highest_thresh)
#
#    description = f"class {label}: {prob}" 
#
#    run["evaluation/predictions"].log(
#        dat, name=pred_label, description=description
#    )

#############################################################
## Speed Test
#############################################################
#start_t = time()
#clf.predict_proba(X)
#end_t = time()
#print(end_t-start_t)

#############################################################
## Save Model 
#############################################################
#xgb.save(clf, os.path.join(model_save_dir, "xgb_classifier"))
dump(clf, os.path.join(model_save_dir, f"{model_type}_classifier"))

pipeline = Pipeline([
    ('zscore', zscore_transform),
    ('pca', pca_obj),
    ('scaler', scaler_obj),
    ('classifier', clf)])
dump(pipeline, os.path.join(model_save_dir, f"{model_type}_pipeline"))

run["model/saved_model"].upload(
        os.path.join(model_save_dir, f"{model_type}_classifier"))

run["model/saved_pipeline"].upload(
        os.path.join(model_save_dir, f"{model_type}_pipeline"))

run.stop()
############################################################
# SHAP analysis 
############################################################
#Xd = xgb.DMatrix(X, label=y)
## make sure the SHAP values add up to marginal predictions
#pred = clf.predict(X, output_margin=True)
#explainer = shap.TreeExplainer(clf)
#shap_values = explainer.shap_values(Xd)
#np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
#
#plt.figure()
#shap.summary_plot(shap_values, X)
#plt.savefig(os.path.join(plot_dir, 'xgboost_shap.png'),
#            dpi=300)
#plt.close()
