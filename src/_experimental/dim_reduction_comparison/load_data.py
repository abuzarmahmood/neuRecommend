"""
Data loading and preprocessing.
"""

import tables
import numpy as np

data_path = '/media/bigdata/projects/neuRecommend/data/final/final_dataset.h5'

def return_data():
    """
    Returns the data as a tuple of numpy arrays.
    """
    with tables.open_file(data_path, 'r') as f:
        pos_data = [x[:] for x in f.root.sorted.pos] 
        neg_data = np.stack([x[:] for x in f.root.sorted.neg]) 
        return pos_data, neg_data
