"""
Use model trained in classify_spikes.py to recommend electrodes to sort
Emphasis on maximizing the precision/recall tradeoff so that no units
are missed but that the recommendations don't have too many false positives.
However, recommendations should err on side of more false positives if needed
so as not to miss electrodes
"""

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
import re
import pandas as pd

import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import multiprocessing

from joblib import dump, load

model_save_dir = '/media/bigdata/projects/neuRecommend/'

def zscore_custom(x):
    return zscore(x,axis=-1)

clf = load(os.path.join(model_save_dir, 'xgb_pipeline'))

## Iterate over sorted recordings
# Use unit_descriptors to check which electrodes have units
waveform_file_path = '/media/bigdata/Abuzar_Data/all_spike_waveform_files.txt'
waveform_file_list = [x.strip() for x in open(waveform_file_path,'r').readlines()]
waveform_basename_list = [x.split('/')[-4] for x in waveform_file_list]
waveform_elec_list = [re.findall('\d+',x.split('/')[-2])[0] \
                            for x in waveform_file_list]
waveform_data_dir = ["/".join(x.split('/')[:-3]) for x in waveform_file_list]

path_frame = pd.DataFrame({
            'path' : waveform_file_list,
            'basename' : waveform_basename_list,
            'data_dir' : waveform_data_dir,
            'electrode' : waveform_elec_list})
path_frame.electrode = path_frame.electrode.astype('int32')

unique_frame = path_frame.drop_duplicates('data_dir')

pos_frames = []
for this_dir in unique_frame.data_dir:
    #this_dir = unique_frame.data_dir.iloc[0]
    dat = ephys_data(this_dir)
    dat.get_region_units()
    pos_electrodes = np.unique([x[1] for x in dat.unit_descriptors])
    this_frame = pd.DataFrame({
                    'data_dir' : this_dir,
                    'electrode' : pos_electrodes,
                    'unit' : True})
    pos_frames.append(this_frame)

fin_pos_frame = pd.concat(pos_frames)

path_frame = path_frame.merge(fin_pos_frame, how = 'outer')
path_frame.fillna(False, inplace=True)
path_frame['electrode_str'] = [f'{x:02}' for x in path_frame.electrode] 

path_frame['array_name'] = path_frame.basename + '_elec' +\
                    path_frame.electrode_str

path_frame.unit.mean()
mean_unit_frac = path_frame.groupby('basename').agg({'unit':'mean'})['unit']

cmap = plt.get_cmap('tab10')
plt.hist(mean_unit_frac, alpha = 0.5, bins = 15, color = cmap(0));
plt.hist(mean_unit_frac, histtype = 'step', bins = 15, color = cmap(0),
        linewidth = 2);
plt.axvline(np.median(mean_unit_frac), color = 'red', alpha = 0.7,
        linewidth = 2, label = 'Median Fraction', linestyle = '--')
plt.legend()
plt.xlabel('Fraction of channels/dataset with neurons')
plt.ylabel('Frequency')
plt.xlim([0,1])
plt.savefig(os.path.join(model_save_dir,'good_channel_frac_dist.png'),
        dpi = 300)
plt.close()
#plt.show()

# For each electrode, calculate class probabilities and save in dataframe
#path_frame['proba'] = None
#for num, path in tqdm(enumerate(path_frame.path)):
#    x = np.load(path)
#    if x.shape[1] == 75:
#        proba = clf.predict_proba(x)
#    elif x.shape[1] == 750:
#        x_temp = x[:,::10]
#        proba = clf.predict_proba(x_temp)
#    else:
#        proba = np.nan
#    path_frame['proba'].loc[num] = proba

# Save frame to avoid having to calculate proba again
#path_frame.to_pickle(os.path.join(model_save_dir,'path_frame.pkl'))
path_frame = pd.read_pickle(os.path.join(model_save_dir,'path_frame.pkl'))

path_frame['wave_count'] = [x.shape[0] for x in path_frame['proba']] 

# We want atleast 2000 waveforms to want to look at an electrodes
# Given that, we want to find the best precision/recall combo
unit_vec = path_frame.unit.to_numpy()
proba_vec = path_frame.proba.to_numpy()

fin_proba_vec = [x[:,1] for x in proba_vec]
pred_spike_count = [np.sum(x>0.18) for x in fin_proba_vec]
rec_channel_vec = np.array(pred_spike_count) > 2000
#precision = precision_score(unit_vec, rec_list) 
recall = recall_score(unit_vec, rec_channel_vec) 
confusion_matrix(unit_vec, rec_channel_vec, normalize='all')

# Given no neuron, how many channels did we correctly throw out
np.mean(rec_channel_vec[unit_vec==False] == False)

#hit_thresh_vec = np.arange(500,2500,500)
hit_thresh_vec = [2000]
metric_frame_list=  []

for hit_thresh in hit_thresh_vec:
    #hit_thresh = 500

    precision_list = []
    recall_list = []
    #true_pos_list = []
    #false_pos_list = []
    #true_neg_list = []
    confusion_list = []
    thresh_vec = np.linspace(0,1,100)

    for thresh in tqdm(thresh_vec):
        #thresh = 0.5
        rec_list = []
        for ind in range(len(unit_vec)):
            #ind = 0
            this_proba = proba_vec[ind][:,1] 
            hit_count = np.sum(this_proba > thresh)
            rec_bool = hit_count > hit_thresh
            rec_list.append(rec_bool)
        rec_list = np.array(rec_list)

        conf_tup = confusion_matrix(unit_vec, rec_list).ravel()
        confusion_list.append(conf_tup)
        #true_pos_list.append(np.mean(rec_list[unit_vec]))
        #false_pos_list.append(np.mean(rec_list[~unit_vec]))
        #true_neg_list.append(np.mean(~rec_list[~unit_vec]))

        precision = precision_score(unit_vec, rec_list) 
        recall = recall_score(unit_vec, rec_list) 
        precision_list.append(precision)
        recall_list.append(recall)

    tn,fp,fn,tp = list(zip(*confusion_list))
    this_frame = pd.DataFrame({
        'hit_thresh' : hit_thresh,
        'precision' : [precision_list],
        'recall' : [recall_list],
        'true_pos' : [tp],#[true_pos_list],
        'false_pos' : [fp],#[false_pos_list],
        'true_neg' :[tn]})# [true_neg_list]})
    metric_frame_list.append(this_frame)

#tn, fp, fn, tp = confusion_matrix(unit_vec, rec_list).ravel()
metric_frame = pd.concat(metric_frame_list)

for num, row in metric_frame.iterrows():
    plt.plot(row.false_pos, row.true_pos,'-x', 
            label = row.hit_thresh, alpha = 0.7)
plt.xlabel('False Pos Rate')
plt.ylabel('True Pos Rate')
plt.legend()
plt.show()

true_pos_thresh = 0.99*np.sum(unit_vec)
true_pos_array = np.stack(metric_frame.true_pos)
true_neg_array = np.stack(metric_frame.true_neg)

true_pos_inds = np.where(true_pos_array >= true_pos_thresh)
cond_true_neg = true_neg_array[true_pos_inds]
max_cond_neg = np.max(cond_true_neg)

for num, row in metric_frame.iterrows():
    plt.plot(row.true_neg, row.true_pos,'-x', 
            label = row.hit_thresh, alpha = 0.7)
plt.xlabel('True Neg Rate')
plt.ylabel('True Pos Rate')
plt.axvline(max_cond_neg)
plt.suptitle(f'{np.round(max_cond_neg,3)} True Negative given '\
        f'{true_pos_thresh} True Positive')
plt.legend()
plt.show()

plt.plot(thresh_vec, precision_list, '-x', label = "Precision")
plt.plot(thresh_vec, recall_list, '-x', label = "Recall")
plt.xlabel('Threshold')
plt.ylabel('Metric')
plt.legend()
plt.show()

plt.plot(recall_list, precision_list,'-x')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
