import tables
import numpy as np
import os
import json

params_dir = '/media/bigdata/projects/neuRecommend/src/create_pipeline/params'


def return_data():
    with open(os.path.join(params_dir, 'path_vars.json'), 'r') as path_file:
        path_vars = json.load(path_file)

    h5_path = path_vars['h5_fin_path']

    # Load equal numbers of waveforms for pos,neg, split into train,test
    # Since positive samples are >> negative, we will subsample from them
    neg_path = '/sorted/neg'
    pos_path = '/sorted/pos'

    neg_waveforms = []
    pos_waveforms = []

    with tables.open_file(h5_path, 'r') as h5:
        for x in h5.iter_nodes(neg_path):
            neg_waveforms.append(x[:])
        for x in h5.iter_nodes(pos_path):
            pos_waveforms.append(x[:])

    neg_waveforms = np.concatenate(neg_waveforms, axis=0)
    pos_waveforms = np.concatenate(pos_waveforms, axis=0)

    neg_label = [0]*neg_waveforms.shape[0]
    pos_label = [1]*pos_waveforms.shape[0]
    fin_labels = np.concatenate([neg_label, pos_label])
    fin_data = np.concatenate([neg_waveforms, pos_waveforms])

    return fin_data, fin_labels
