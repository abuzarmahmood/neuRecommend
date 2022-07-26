import tables
import numpy as np
from tqdm import tqdm
import os
import matplotlib
#matplotlib.use('TKAgg')
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

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import multiprocessing

from joblib import dump, load


h5_path = '/media/bigdata/projects/neuRecommend/test_data/sorted/sorted_waveforms.h5'
model_save_dir = '/media/bigdata/projects/neuRecommend/'

def imshow(array):
    plt.imshow(array, interpolation = 'nearest', aspect='auto')

# Load equal numbers of waveforms for pos,neg, split into train,test
# Since positive samples are >> negative, we will subsample from them
neg_path = '/sorted/neg'
pos_path = '/sorted/pos'

neg_waveforms = []
pos_waveforms = []

#with tables.open_file(h5_path,'r') as h5:
h5 = tables.open_file(h5_path,'r')
for x in h5.iter_nodes(neg_path):
    neg_waveforms.append(x[:])

neg_waveforms = np.concatenate(neg_waveforms,axis=0)

# pos_waveforms needs to be of length 75, or 750 that can be downsampled
pos_node_list = list(h5.iter_nodes(pos_path))
# Waveforms with same length as neg_waveforms
pos_matched_units = [x for x in pos_node_list \
        if x.shape[1] == neg_waveforms.shape[1]]
waveforms_per_unit = neg_waveforms.shape[0]//len(pos_matched_units)

#with tables.open_file(h5_path,'r') as h5:
for x in pos_matched_units:
    ind = np.min([x.shape[0], waveforms_per_unit])
    pos_waveforms.append(x[:ind,:])
pos_waveforms = np.concatenate(pos_waveforms,axis=0)
h5.close()

neg_label = [0]*neg_waveforms.shape[0]
pos_label = [1]*pos_waveforms.shape[0]
fin_labels = np.concatenate([neg_label, pos_label])

############################################################
## Train classifier
############################################################
def zscore_custom(x):
    return zscore(x,axis=-1)
zscore_transform = FunctionTransformer(zscore_custom)
                

fin_data = np.concatenate([neg_waveforms, pos_waveforms]) 
#zscore_fin_data = zscore(fin_data, axis=-1)
zscore_fin_data2 = zscore_transform.transform(fin_data)
pca_obj = pca(n_components = 10).fit(zscore_fin_data2[::1000])
print(f'Explained variance : {np.sum(pca_obj.explained_variance_ratio_)}')
pca_data = pca_obj.transform(zscore_fin_data2)

scaler_obj = StandardScaler().fit(pca_data)
X = scaler_obj.transform(pca_data)
trim_factor = 1
X = X[::trim_factor]
y = fin_labels[::trim_factor]

#imshow(fin_data[::1000]);plt.show()
#imshow(zscore_fin_data[::1000]);plt.show()
#imshow(pca_data[::1000]);plt.show()
#imshow(X[::1000]);plt.show()

#X_train, X_test, y_train, y_test = train_test_split(
#    X, fin_labels, test_size=0.4, random_state=42
#)

X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.5, random_state=1)

X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

#xgb_model = xgb.XGBClassifier(n_jobs=multiprocessing.cpu_count() // 2)
#clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
#               'n_estimators': [50, 100, 200]}, verbose=1, n_jobs=2)
#clf.fit(X, y)
#print(clf.best_score_)
#print(clf.best_params_)
#
## Write out best_params to json
optim_params_path = os.path.join(model_save_dir,'optim_params.json')
#with open(optim_params_path,'w') as outfile:
#    json.dump(clf.best_params_, outfile, indent = 4)

with open(optim_params_path,'r') as outfile:
    best_params = json.load(outfile)

#
#clf = xgb.XGBClassifier(**clf.best_params_)
clf = xgb.XGBClassifier(**best_params)

#clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)

############################################################
## Titrate decision boundaries 
############################################################
val_proba = clf.predict_proba(X_val)[:,1]

all_proba = clf.predict_proba(X)[:,1]


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

thresh_vec = np.linspace(0,1,100)
true_mean_val = []
for this_thresh in thresh_vec:
    pred_pos = val_proba >= this_thresh
    pred_pos = pred_pos*1 
    pred_pos_mean = np.mean(pred_pos[np.where(y_val)[0]])
    true_mean_val.append(pred_pos_mean)
true_mean_val = np.array(true_mean_val)


false_num, proba = np.histogram(val_proba[np.where(y_val==0)[0]], bins = 100) 
cumu_false = np.cumsum(false_num)
scaled_cumu_false = cumu_false/np.max(cumu_false)
        
true_mean_thresh_inds = true_mean_val >= wanted_thresh
highest_thresh = np.max(thresh_vec[true_mean_thresh_inds])
best_false = scaled_cumu_false[np.argmin(np.abs(proba-highest_thresh))]


############################################################
# Plot distribution of predicted probabilities
############################################################
left_str = '1% Spikes' + '\n' + '93% Noise'
right_str = '99% Spikes' + '\n' + '7% Noise'

plt.hist(val_proba[np.where(y_val)[0]], bins = 50, 
        alpha = 0.5, label = 'True Spike')
#plt.yscale('log')
plt.hist(val_proba[np.where(y_val==0)[0]], 
        alpha = 0.5, bins = 50, label = 'True Noise')
plt.xlabel('Spike probability')
plt.ylabel('Frequency (Log Scale)')
plt.axvline(highest_thresh, 
        color = 'red', alpha = 0.7, linewidth = 2,
        label = f'Threshold for {wanted_thresh} Recall')
plt.text(0.05, 0.6, left_str, transform = plt.gca().transAxes)
plt.text(0.25, 0.6, right_str, transform = plt.gca().transAxes)
plt.title('Distribution of classification probabilities')
plt.legend()
plt.savefig(os.path.join(model_save_dir,'spike_classification_dist.png'),
        dpi = 300)
plt.close()
#plt.show()

############################################################
# Representative plots of spikes and noise at different probs 
############################################################
wanted_probs = np.linspace(0,1,6)
prob_frame = pd.DataFrame(
        dict(
            label = y,
            prob = all_proba 
            )
        )

nrn_inds = []
noise_inds = []
for this_prob in wanted_probs:
    spike_val = (prob_frame['prob'][prob_frame.label == 1] - this_prob)\
            .abs().argsort()[:10].values
    noise_val = (prob_frame['prob'][prob_frame.label == 0] - this_prob)\
            .abs().argsort()[:10].values
    nrn_inds.append(spike_val)
    noise_inds.append(noise_val)

fig,ax = plt.subplots(1,len(wanted_probs), sharey = True,
        figsize = (13,2))
for num in range(len(wanted_probs)):
    #ax[0,num].plot(fin_data[nrn_inds[num]])
    #ax[1,num].plot(fin_data[noise_inds[num]])
    this_dat = zscore(fin_data[noise_inds[num]],axis=-1)
    flip_bool = np.vectorize(np.int)(np.sign(this_dat[:,30]))
    this_dat = np.stack([x*-1 if this_bool==1 else x \
            for x,this_bool in zip(this_dat, flip_bool)])
    ax[num].plot(this_dat.T,
            color = 'k', alpha = 0.5)
    ax[num].set_title(f'Prob : {np.round(wanted_probs[num],1)}')
    ax[num].set_xticklabels([])
    ax[num].set_xticks([])
    ax[num].set_yticklabels([])
    ax[num].set_yticks([])
plt.savefig(os.path.join(model_save_dir,'spike_classification_examples.png'),
        dpi = 300)
plt.close()
#plt.show()

#plt.plot(thresh_vec, true_mean_val, label = 'True fraction')
#plt.scatter(thresh_vec[true_mean_thresh_inds], 
#        true_mean_val[true_mean_thresh_inds],
#        marker = 'x', color = 'orange', label = 'Pass Thresh')
#plt.plot(proba[:-1], scaled_cumu_false, label = "False fraction")
#plt.axvline(highest_thresh, color = 'red', label = 'Highest Thresh')
#plt.xlabel('Threshold')
#plt.ylabel('Fraction of True Positives detected')
#plt.suptitle(f'{np.round(best_false,3)} rejected @ {wanted_thresh} true accepted') 
#plt.legend()
#plt.show()

## Test on test-set using highest thresh
test_proba = clf.predict_proba(X_test)[:,1]
pred_pos = test_proba >= highest_thresh 
pred_pos = pred_pos*1 
pred_false = test_proba < highest_thresh
true_recovered = np.mean(pred_pos[np.where(y_test)[0]])
false_detected = np.mean(pred_false[np.where(y_test ==0)[0]])
print(f'{np.round(true_recovered,3)} True recovered' + '\n' +\
        f'{np.round(false_detected,3)} False detected')

############################################################
## Speed Test
############################################################
start_t = time()
clf.predict_proba(X)
end_t = time()
print(end_t-start_t)

#xgb.save(clf, os.path.join(model_save_dir, "xgb_classifier"))
dump(clf, os.path.join(model_save_dir, "xgb_classifier"))

pipeline = Pipeline([
                ('zscore', zscore_transform),
                ('pca', pca_obj),
                ('scaler', scaler_obj), 
                ('svc', clf)])
dump(pipeline, os.path.join(model_save_dir, "xgb_pipeline"))
