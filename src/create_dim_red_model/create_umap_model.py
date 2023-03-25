import feature_engineering_pipeline as fep
from return_data import return_data
import numpy as np
import os
import pylab as plt
import json
from sklearn.decomposition import PCA as pca
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from umap.parametric_umap import ParametricUMAP
from joblib import dump

import sys
sys.path.append('../create_pipeline')

with open('./params/path_vars.json', 'r') as path_file:
    path_vars = json.load(path_file)

############################################################
# Feature Engineering
############################################################

X_raw, y = return_data()

#energy_raw = fep.EnergyFeature().fit_transform(X_raw)
#amplitude_raw = fep.AmpFeature().fit_transform(X_raw)

# Log transform to make values more manageable
#energy = np.log(energy_raw)
#amplitude = np.log(amplitude_raw)

# Zscore data
zscore_X_raw = fep.zscore_transform.transform(X_raw)

# PCA: Reduce to components that preserve 95% of the variance
max_components = 15
pca_obj = pca(n_components=max_components).fit(zscore_X_raw[::100])
wanted_components = np.where(
    np.cumsum(pca_obj.explained_variance_ratio_) > 0.95)[0][0]
pca_obj = pca(n_components=wanted_components).fit(zscore_X_raw[::100])
pca_data = pca_obj.transform(zscore_X_raw)

# Create Pipeline
log_transform = FunctionTransformer(np.log, validate=True)

# Energy and amplitude are almost perfectly correlated
# Take only amplitude
# energy_pipeline = Pipeline(
#        steps=[
#            ('energy', fep.EnergyFeature()),
#            ('log', log_transform),
#        ]
#    )

amplitude_pipeline = Pipeline(
    steps=[
        ('amplitude', fep.AmpFeature()),
        ('log', log_transform),
    ]
)

pca_pipeline = Pipeline(
    steps=[
        ('zscore', fep.zscore_transform),
        ('pca', pca_obj),
    ]
)

collect_feature_pipeline = FeatureUnion(
    n_jobs=1,
    transformer_list=[
        # ('energy', energy_pipeline),
        ('amplitude', amplitude_pipeline),
        ('pca_features', pca_pipeline),
    ]
)

all_features = collect_feature_pipeline.transform(X_raw)
# Final scaling also has to stay constant
scaler = StandardScaler().fit(all_features)

feature_pipeline = Pipeline(
    steps=[
        ('get_features', collect_feature_pipeline),
        ('scale_features', scaler)
    ]
)

############################################################
# Write out pipeline
############################################################

dump(feature_pipeline,
     os.path.join(
         path_vars['model_save_dir'],
         "umap_feature_pipeline.dump"
     )
     )

# Concatenate features
#cat_X = np.concatenate([energy, amplitude, pca_data], axis=1)
#
# Standardize data
#scaler = StandardScaler().fit(cat_X)
#scaled_X = scaler.transform(cat_X)

scaled_X = feature_pipeline.transform(X_raw)

#cor_mat = np.corrcoef(scaled_X_pipeline.T, scaled_X.T)
# plt.matshow(cor_mat);plt.show()

## Plot X and y together
#fig, ax = plt.subplots(1, 2)
#im = ax[0].imshow(scaled_X, aspect='auto',
#                  interpolation='none', cmap='viridis')
#plt.colorbar(im, ax=ax[0])
#ax[1].plot(y, np.arange(len(y)))
#plt.show()

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
    path_vars['model_save_dir'], 'umap_model.pkl'))
