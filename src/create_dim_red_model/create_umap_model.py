import os
import json
from umap.parametric_umap import ParametricUMAP
from joblib import load

import sys
sys.path.append('../create_pipeline')
from return_data import return_data
from feature_engineering_pipeline import * 

params_dir = '../create_pipeline/params'
with open(os.path.join(params_dir, 'path_vars.json'), 'r') as path_file:
    path_vars = json.load(path_file)
with open(os.path.join(params_dir, 'data_params.json'), 'r') as path_file:
    data_params = json.load(path_file)
zero_ind = data_params['zero_ind']

model_save_dir = path_vars['model_save_dir']
feature_pipeline_path = path_vars['feature_pipeline_path']

############################################################
# Feature Engineering
############################################################

# Even though Amplitude and Energy are almost perfectly
# correlated, apparently having both of them helps the 
# XGBoost model. So we keep them both.
# Refer to Neptune.AI logs for more details.
X_raw, y = return_data()
feature_pipeline = load(feature_pipeline_path)
scaled_X = feature_pipeline.transform(X_raw)

polarity = np.sign(AmpFeature(zero_ind).transform(X_raw)).flatten()

############################################################
# Train UMAP
############################################################
# Train 2D UMAPs for each polarity separately
n_components = 2

for this_polarity in [1, -1]:
    print('Training UMAP for polarity: {}'.format(this_polarity))
    parametric_umap = ParametricUMAP(
        n_components=n_components,
        verbose=True,
    )

    trimming = 10
    this_data = scaled_X[polarity == this_polarity]
    parametric_umap.fit(this_data[::trimming])

    transformed_data = parametric_umap.transform(scaled_X)

    # Save model
    parametric_umap.save(os.path.join(
        path_vars['model_save_dir'], 'umap_model_polarity_{}'.format(this_polarity)))
