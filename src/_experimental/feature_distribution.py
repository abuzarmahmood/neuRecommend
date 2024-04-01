import tables
import os
import numpy as np
from joblib import load
import seaborn as sns
import pandas as pd
import pylab as plt

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
pipeline = load(feature_pipeline_path)

pos_features = pipeline.transform(pos_dat)
feature_labels = [f'ft{x}' for x in range(pos_features.shape[1])]
pos_feature_frame = pd.DataFrame(
        pos_features, 
        columns = feature_labels
        )
pos_feature_frame['label'] = 'pos'

sns.pairplot(pos_feature_frame, kind = 'hist');plt.show()

neg_features = pipeline.transform(neg_dat)
neg_feature_frame = pd.DataFrame(
        neg_features, 
        columns = feature_labels
        )
neg_feature_frame['label'] = 'neg'

fin_frame = pd.concat([pos_feature_frame, neg_feature_frame])

sns.pairplot(
    fin_frame.iloc[::100,:],
       hue = 'label',
       kind = 'hist');
plt.show()
