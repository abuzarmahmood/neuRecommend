from return_data import return_data
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import zscore
import numpy as np
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import FunctionTransformer
from joblib import dump, load
import os
import json


class EnergyFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y = None ):
        return self
    def transform(self, X, y=None):
        energy = np.sqrt(np.sum(X**2, axis = -1))/X.shape[-1]
        energy = energy[:,np.newaxis]
        return energy

class AmpFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y = None ):
        return self
    def transform(self, X, y=None):
        amplitude = np.max(np.abs(X), axis=-1)
        amplitude = amplitude[:,np.newaxis]
        return amplitude 

def zscore_custom(x):
    return zscore(x,axis=-1)

if __name__ == "__main__":
    X_raw,y = return_data()
    zscore_transform = FunctionTransformer(zscore_custom)

    # We have to store the same PCA object to use later
    # Otherwise the features won't make sense to the classifier
    pca_components = 8
    zscore_X_raw = zscore_transform.transform(X_raw)
    pca_obj = pca(n_components=pca_components).fit(zscore_X_raw[::100])

    pca_pipeline = Pipeline(
            steps = [
                ('zscore',zscore_transform),
                ('pca',pca_obj),
                ]
            )

    feature_pipeline = FeatureUnion(
            n_jobs = -1,
            transformer_list = [
                ('pca_features', pca_pipeline),
                ('energy', EnergyFeature()),
                ('amplitude', AmpFeature()),
                ]
            )

    ############################################################
    # Write out pipeline
    ############################################################

    with open('path_vars.json','r') as path_file:
        path_vars = json.load(path_file)
    model_save_dir = path_vars['model_save_dir']

    dump(feature_pipeline, 
            os.path.join(
                model_save_dir, 
                f"feature_engineering_pipeline.dump"
                )
            )
