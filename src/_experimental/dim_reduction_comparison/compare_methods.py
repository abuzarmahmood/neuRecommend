import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.decomposition import PCA
# from sklearn.neural_network import MLPRegressor
from keras_autoencoder import autoencoder_class
import umap
from umap.parametric_umap import ParametricUMAP
import sys
from time import time
from tqdm import trange
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
        """
        Initialize the class with the data and labels

        data: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples, )
        """
        self.data = self.keep_pca_dims(self.zscore(data))
        self.labels = labels
        self.data = self.data[::10]
        self.labels = self.labels[::10]
        self.split_data()


    # ============================================================
    # Preprocessing
    def zscore(self, data):
        """
        Zscore individual datapoints
        """
        return (data - np.mean(data, axis=1)[:, None])/np.std(data, axis=1)[:, None]

    def keep_pca_dims(self, data, explained_variance_threshold=0.95):
        """
        Return transformed data with only the principal components that explain 
        a certain amount of variance
        """
        pca = PCA().fit(data)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= explained_variance_threshold) + 1
        pca_data = pca.transform(data)[:, :n_components]
        return pca_data

    def split_data(self, test_size=0.5):
        """
        Split data into test and train sets and store as attributes
        """
        if 'data' not in dir(self):
            raise AttributeError('No data attribute found. Please initialize the class with data and labels')
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                    train_test_split(self.data, self.labels, test_size=test_size)

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
        autoencoder = autoencoder_class(self.data.shape[1], n_components)
        return autoencoder

    def umap(self, n_components=2):
        """
        Uniform Manifold Approximation and Projection.
        """
        print(f'Creating UMAP with {n_components} components')
        reducer = umap.UMAP(
                n_components=n_components,
                verbose=True)
        return reducer

    def parametric_umap(self, n_components=2):
        """
        Parametric UMAP
        """
        print(f'Creating Parametric UMAP with {n_components} components')
        reducer = ParametricUMAP(
                n_components=n_components,
                parametric_reconstruction=True,
                )
        return reducer

    # ============================================================

    # Define Methods
    # ============================================================
    def fit_method(self, model):
        """
        Fit the model to the data and record the time
        """
        print(f'Fit method: {model}')
        print(f'X_train shape: {self.X_train.shape}')
        start = time()
        if not 'MLPRegressor' in model.__repr__(): 
            model = model.fit(self.X_train)
        else:
            model = model.fit(self.X_train, self.X_train)
        end = time()
        return model, end - start

    def transform_method(self, model):
        """
        Transform the data with the model and record the time
        """
        print(f'Transform method: {model}')
        print(f'X_test shape: {self.X_test.shape}')
        start = time()
        transformed_data = model.transform(self.X_test)
        end = time()
        return transformed_data, end - start

    def reconstruct_data(self, fitted_model, transformed_data):
        """
        Reconstruct the data from the transformed data given the fitted model
        """
        print(f'Reconstruct data: {fitted_model}')
        return fitted_model.inverse_transform(transformed_data)

    def calculate_variance_explained(
            self,
            reconstructed_data,
            fitted_model):
        """
        Calculate the variance explained by the transformed data
        """
        print(f'Calculating variance for model: {fitted_model}')
        return explained_variance_score(self.X_test, reconstructed_data)

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
        elif model_name == 'parametric_umap':
            model = self.parametric_umap(n_components=n_components)
        else:
            print('Model not found')
            return

        fitted_model, fit_time = self.fit_method(model)
        transformed_data, transform_time = self.transform_method(fitted_model)
        variance = self.calculate_variance_explained(
            self.reconstruct_data(fitted_model, transformed_data),
            fitted_model)
        return fit_time, transform_time, variance

    def iterate_over_components(
            self,
            min_components,
            max_components,
            model_name):
        """
        Run <run_methods_for_n_components> for all the components in the range
        """
        fit_times = []
        transform_times = []
        variance_explained = []
        for n_components in trange(min_components, max_components):
            (
                fit_time,
                transform_time,
                variance) = self.run_methods_for_n_components(
                n_components,
                model_name
            )
            fit_times.append(fit_time)
            transform_times.append(transform_time)
            variance_explained.append(variance)
        return fit_times, transform_times, variance_explained
