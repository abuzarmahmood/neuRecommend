"""
https://docs.neptune.ai/getting-started/how-to-add-neptune-to-your-code
"""

import numpy as np
import os
import pylab as plt
from time import time
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import xgboost as xgb
import shap

from joblib import dump, load

from return_data import return_data
from feature_engineering_pipeline import *

import neptune.new as neptune

############################################################
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################
with open('./params/path_vars.json','r') as path_file:
    path_vars = json.load(path_file)

h5_path = path_vars['h5_fin_path']
h5_dir = os.path.dirname(h5_path)
model_save_dir = path_vars['model_save_dir']
plot_dir = path_vars['plot_dir']
feature_pipeline_path = path_vars['feature_pipeline_path']

############################################################
# Train classifier
############################################################

# It doesn't matter whether we transform first and split later, or vice versa,
# because all steps in the pipeline which are dependent on dataset 
# have fixed parameters

X_raw,y = return_data()
feature_pipeline = load(feature_pipeline_path)
X = feature_pipeline.transform(X_raw)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, random_state=1)

X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.25, random_state=1)

########################################
optim_params_path = '../../model/optim_params.json'
with open(optim_params_path, 'r') as outfile:
    best_params = json.load(outfile)

clf = xgb.XGBClassifier(**best_params, n_jobs = 1)

train_start = time()
clf.fit(X_train, y_train)
train_end = time()
fit_time = train_end - train_start

pred_start = time()
train_score = clf.score(X_train, y_train)
pred_end = time()
pred_time = pred_end-pred_start

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

thresh_out_path = os.path.join(model_save_dir, 'proba_threshold.json')
with open(thresh_out_path, 'w') as outfile:
    json.dump(dict(threshold = highest_thresh), outfile)

fin_val_pred = val_proba > highest_thresh
labels = ['true_neg','false_neg','false_pos','true_pos']
conf_mat = confusion_matrix(y_val, fin_val_pred, normalize = 'true')
conf_dict = dict(zip(labels, np.round(conf_mat.ravel(),4)))
print(conf_dict)

thresh_accuracy = np.mean((val_proba > highest_thresh) == y_val)

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
plt.text(highest_thresh -0.2, 0.6, left_str, transform=plt.gca().transAxes)
plt.text(highest_thresh +0.05, 0.6, right_str, transform=plt.gca().transAxes)
plt.title('Distribution of classification probabilities')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'spike_classification_dist.png'),
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

inds = []
for this_prob in wanted_probs:
    val = (prob_frame['prob'] - this_prob)\
        .abs().argsort()[:10].values
    inds.append(val)

fig, ax = plt.subplots(1, len(wanted_probs), sharey=True,
                       figsize=(13, 2))
for num in range(len(wanted_probs)):
    this_dat = zscore(X_raw[inds[num]], axis=-1)
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
plt.savefig(os.path.join(plot_dir, 'spike_classification_examples.png'),
            dpi=300)
plt.close()
# plt.show()

#############################################################
## Save Model 
#############################################################
pipeline = Pipeline([
    ('feature_engineering', feature_pipeline),
    ('classifier', clf)])
dump(clf, os.path.join(model_save_dir, f"xgboost_classifier.dump"))
dump(pipeline, os.path.join(model_save_dir, f"xgboost_full_pipeline.dump"))

############################################################
# SHAP analysis 
############################################################
feature_labels = [
        [f'pca{i}' for i in range(X.shape[1]-2)],
        ['energy'],
        ['amplitude']
        ]
feature_labels = [x for y in feature_labels for x in y]
#X_frame = pd.DataFrame(X, columns = feature_labels)
#
#Xd = xgb.DMatrix(X_frame, label=y)
#pred = clf.predict(X_frame, output_margin=True)
#explainer = shap.TreeExplainer(clf)
#shap_values = explainer(Xd)
#
#plt.figure()
#shap.summary_plot(shap_values.values[::100], X_frame.iloc[::100],
#        show = False)
#plt.savefig(os.path.join(plot_dir, f'xgboost_shap_summary.png'),
#            dpi=300)
#plt.close()
#
#plt.figure()
#shap.plots.bar(shap_values, max_display = len(feature_labels),
#        show = False)
#ax = plt.gca()
#ticks = ax.get_yticks()[:len(feature_labels)]
#labels = [x._text for x in ax.get_yticklabels()[:len(feature_labels)]]
#digits = [x.split(' ')[-1] for x in labels] 
#label_ind = [int(x) for x in digits if x.isdigit()]
#sorted_labels = [feature_labels[x] for x in label_ind]
#plt.yticks(ticks, labels = sorted_labels)
#plt.tight_layout()
#plt.savefig(os.path.join(plot_dir, f'xgboost_shap_bar.png'),
#            dpi=300) 
#plt.close()

#############################################################
## Log to neptune 
#############################################################
with open('./params/neptune_params.json','r') as path_file:
    neptune_params = json.load(path_file)

#model = neptune.init_model_version(
#    model = 'WAV-CLF',
#    project= neptune_params['project'],
#    api_token = neptune_params['api_token']
#    )
model = neptune.init_run(
    project= neptune_params['project'],
    api_token = neptune_params['api_token']
    )

model["data/train_dataset"].track_files(h5_path)
model["data/train_dataset_metadata"].upload(
    os.path.join(h5_dir, 'fin_data_metadata.csv'))
model["data/train_dataset_summary_stats"].upload(
    os.path.join(h5_dir, 'fin_data_summary_stats.json'))
model["data/feature_labels"] = feature_labels
model["model/parameters"] = best_params
model['model/type'] = 'xgboost' 
model["model/fit_time"] = np.round(fit_time, 3) 
model["model/train_set_size"] = X_train.shape 
model["model/train_set_pred_time"] = np.round(pred_time, 3) 
model["model/parameters/final_threshold"] = highest_thresh
model["evaluation/final_accuracy"] = thresh_accuracy 
model["evaluation/final_confusion_matrix"] = conf_dict
model["model/feature_pipeline"].upload(feature_pipeline_path)
model["model/saved_model"].upload(
    os.path.join(model_save_dir, f"xgboost_classifier.dump"))
model["model/saved_full_pipeline"].upload(
    os.path.join(model_save_dir, f"xgboost_full_pipeline.dump"))
#model['model/shap_summary'].upload(
#    os.path.join(plot_dir, f'xgboost_shap_summary.png') 
#    )
#model['model/shap_bar'].upload(
#    os.path.join(plot_dir, f'xgboost_shap_bar.png')        
#    )
model["notes"] = "Single core fit and predict"
model.stop()
