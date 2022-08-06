"""
Create final dataset for training such that pos and neg samples are balanced
and each set includes maximum diversity of clusters
Max total size limit = 2GB
"""
import tables
import numpy as np
from tqdm import tqdm
import os
import json

with open('path_vars.json','r') as path_file:
    path_vars = json.load(path_file)

h5_path = path_vars['h5_path']
h5_dir = os.path.dirname(h5_path)
model_save_dir = path_vars['model_save_dir']

# Load equal numbers of waveforms for pos,neg, split into train,test
# Since positive samples are >> negative, we will subsample from them
neg_path = '/sorted/neg'
pos_path = '/sorted/pos'

h5 = tables.open_file(h5_path, 'r')
allowed_lens = [75,750]
neg_node_list = list(h5.iter_nodes(neg_path))
neg_total = np.sum([x.shape[0] for x in neg_node_list \
        if x.shape[1] in allowed_lens])

# pos_waveforms needs to be of length 75, or 750 that can be downsampled
pos_node_list = list(h5.iter_nodes(pos_path))
pos_node_list = [x for x in pos_node_list \
        if x.shape[1] in allowed_lens]
# Take samples from all available units to maintain diversity
waveforms_per_unit = neg_total//len(pos_node_list)

############################################################
# Create new h5 file
# Copy over negative samples
# Copy over waveforms_per_unit number of positive samples / neuron

hf5_out = tables.open_file(os.path.join(h5_dir, 'final_dataset.h5')
                    , 'w', title = 'sorted_waveforms')
hf5_out.create_group('/sorted', 'pos', createparents=True)
hf5_out.create_group('/sorted', 'neg', createparents=True)

for this_node in tqdm(pos_node_list):
    save_path = pos_path
    if this_node.shape[1] == 750:
        trim = 10
    else:
        trim = 1
    hf5_out.create_array(
            save_path,
            this_node.name,
            this_node[:waveforms_per_unit, ::trim]
            )

for this_node in tqdm(neg_node_list):
    save_path = neg_path
    if this_node.shape[1] == 750:
        trim = 10
    else:
        trim = 1
    hf5_out.create_array(
            save_path,
            this_node.name,
            this_node[:,::trim]
            )

hf5_out.flush()
hf5_out.close()
