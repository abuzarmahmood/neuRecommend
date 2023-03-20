import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import explained_variance_score
from sklearn.decomposition import PCA
import umap
import sys
from time import time
from tqdm import tqdm, trange
load_data_path = '/media/bigdata/projects/neuRecommend/src/create_pipeline'
sys.path.append(load_data_path)

class dim_red_comparison():
    """
    Methods to compare dimensionality reduction methods.
        1- PCA
        2- Autoencoders
        3- UMAP
        4- Convolutional Autoencoders
    On the following metrics:
        - training and transformation times
        - variance explained
        - and cross-validation classification accuracy
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    # Define Models
    # ============================================================
    def pca(self, n_components=2):
        """
        Principal component analysis via sklearn.
        """
        print(f'Creating PCA with {n_components} components')
        pca = PCA(n_components=n_components)
        return pca

    def autoencoder(self, n_components=2):
        """
        Autoencoder constructed from sklearn.neural_network.MLPRegressor
        """
        print(f'Creating Autoencoder with {n_components} components')
        # Define the input layer
        input_layer = Input(shape=(self.data.shape[1],))

        # Define the encoder
        encoded = Dense(1000, activation='relu')(input_layer)
        encoded = Dense(500, activation='relu')(encoded)
        encoded = Dense(250, activation='relu')(encoded)
        encoded = Dense(125, activation='relu')(encoded)
        encoded = Dense(50, activation='relu')(encoded)
        encoded = Dense(n_components, activation='relu')(encoded)

        # Define the decoder
        decoded = Dense(50, activation='relu')(encoded)
        decoded = Dense(125, activation='relu')(decoded)
        decoded = Dense(250, activation='relu')(decoded)
        decoded = Dense(500, activation='relu')(decoded)
        decoded = Dense(1000, activation='relu')(decoded)
        decoded = Dense(self.data.shape[1], activation='relu')(decoded)

        # Define the model
        autoencoder = Model(input_layer, decoded)

        # Compile the model
        autoencoder.compile(optimizer='adam', loss='mse')

        # Define the encoder model
        encoder = Model(input_layer, encoded)

        # Fit the model
        autoencoder.fit(self.data, self.data,
                        epochs=100,
                        batch_size=256,
                        shuffle=True,
                        validation_split=0.2,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

        return encoder

    def umap(self, n_components=2):
        """
        Uniform Manifold Approximation and Projection.
        """
        print(f'Creating UMAP with {n_components} components')
        reducer = umap.UMAP(n_components=n_components)
        return reducer

    # ============================================================

    # Define Methods
    # ============================================================
    def fit_method(self, model):
        """
        Fit the model to the data and record the time
        """
        print(f'Fit method: {model}')
        start = time()
        model.fit(self.data)
        end = time()
        return model, end - start

    def transform_method(self, model):
        """
        Transform the data with the model and record the time
        """
        print(f'Transform method: {model}')
        start = time()
        transformed_data = model.transform(self.data)
        end = time()
        return transformed_data, end - start

    def reconstruct_data(self, fitted_model, transformed_data):
        """
        Reconstruct the data from the transformed data given the fitted model
        """
        print(f'Reconstruct data: {fitted_model}')
        return fitted_model.inverse_transform(transformed_data)

    def calculate_variance_explained(self, original_data, transformed_data, fitted_model):
        """
        Calculate the variance explained by the transformed data
        """
        print(f'Calculating variance for model: {fitted_model}')
        return explained_variance_score(original_data, transformed_data)

    def run_methods_for_n_components(self, n_components, model_name):
        """
        Run the methods for a given number of components given model_name
        """
        if model_name == 'pca':
            model = self.pca(n_components=n_components)
        elif model_name == 'autoencoder':
            model = self.autoencoder(n_components=n_components)
        elif model_name == 'umap':
            model = self.umap(n_components=n_components)
        else:
            print('Model not found')
            return

        fitted_model, fit_time = self.fit_method(model)
        transformed_data, transform_time = self.transform_method(fitted_model)
        variance = self.calculate_variance_explained(
                    self.data, 
                    self.reconstruct_data(fitted_model, transformed_data),
                    fitted_model)
        return fit_time, transform_time, variance

    def iterate_over_components(self, min_components, max_components, model_name):
        """
        Run <run_methods_for_n_components> for all the components in the range
        """
        fit_times = []
        transform_times = []
        variance_explained = []
        for n_components in trange(min_components, max_components):
            fit_time, transform_time, variance = self.run_methods_for_n_components(
                n_components, model_name)
            fit_times.append(fit_time)
            transform_times.append(transform_time)
            variance_explained.append(variance)
        return fit_times, transform_times, variance_explained
