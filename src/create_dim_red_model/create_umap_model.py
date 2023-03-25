import os
import json
from umap.parametric_umap import ParametricUMAP
from joblib import load

import sys
sys.path.append('../create_pipeline')
from return_data import return_data
from feature_engineering_pipeline import * 

with open('../create_pipeline/params/path_vars.json', 'r') as path_file:
    path_vars = json.load(path_file)
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

############################################################
# Train UMAP
############################################################
n_components = 3

parametric_umap = ParametricUMAP(
    n_components=n_components,
    verbose=True,
)

trimming = 10
parametric_umap.fit(scaled_X[::trimming])

transformed_data = parametric_umap.transform(scaled_X)

# Save model
parametric_umap.save(os.path.join(
    path_vars['model_save_dir'], 'umap_model'))
