"""
Pipeline:
    1 - Extract Energy + Amplitude from raw data
    2 - Zscore data and perform PCA
    3 - Scale all features

Static vs Dynamic Steps:
    - Energy : Dynamic
    - Amplitude : Dynamic
    - Zscore : Dynamic
        - This isn't an issue because EACH WAVEFORM is separately
            getting zscored
    - PCA : Static
        - Each new dataset will give different PCA components,
            therefore the PCA transform must remain fixed
    - StandardScaler : Static
        - Same deal as PCA. Scale of scaling must remain the same 
"""


from return_data import return_data
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import zscore
import numpy as np
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import FunctionTransformer, StandardScaler
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
    pca_components = 3
    zscore_X_raw = zscore_transform.transform(X_raw)
    pca_obj = pca(n_components=pca_components).fit(zscore_X_raw[::100])


    pca_pipeline = Pipeline(
            steps = [
                ('zscore',zscore_transform),
                ('pca',pca_obj),
                ]
            )

    collect_feature_pipeline = FeatureUnion(
            n_jobs = -1,
            transformer_list = [
                ('pca_features', pca_pipeline),
                ('energy', EnergyFeature()),
                ('amplitude', AmpFeature()),
                ]
            )

    all_features = collect_feature_pipeline.transform(X_raw)
    # Final scaling also has to stay constant
    scaler = StandardScaler().fit(all_features)

    feature_pipeline = Pipeline(
            steps = [
                ('get_features', collect_feature_pipeline),
                ('scale_features', scaler)
                ]
            )

    ############################################################
    # Write out pipeline
    ############################################################

    with open('./params/path_vars.json','r') as path_file:
        path_vars = json.load(path_file)
    model_save_dir = path_vars['model_save_dir']

    dump(feature_pipeline, 
            os.path.join(
                model_save_dir, 
                f"feature_engineering_pipeline.dump"
                )
            )
