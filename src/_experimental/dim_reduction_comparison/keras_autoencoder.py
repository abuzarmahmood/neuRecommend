from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input, Dense
from keras import Sequential

class autoencoder_class():
    """
    Autoencoder constructed from sklearn.neural_network.MLPRegressor
    """

    def __init__(self, input_shape, n_components=2):
        self.input_shape = input_shape
        self.model = self.create_model(input_shape, n_components)

    def create_model(self, input_shape, n_components):
        """
        Generate autoencoder according to input shape and save
        both autoencoder and encoder models
        """
        # Define the input layer
        input_layer = Input(shape=(input_shape,))

        # Define the encoder
        # encoded = Dense(500, activation='relu')(input_layer)
        # encoded = Dense(250, activation='relu')(encoded)
        encoded = Dense(125, activation='relu')(input_layer)
        encoded = Dense(50, activation='relu')(encoded)
        encoded = Dense(n_components, activation='relu')(encoded)

        # Define the decoder
        decoded = Dense(50, activation='relu')(encoded)
        decoded = Dense(125, activation='relu')(decoded)
        # decoded = Dense(250, activation='relu')(decoded)
        # decoded = Dense(500, activation='relu')(decoded)
        decoded = Dense(input_shape, activation='relu')(decoded)

        # Define the model
        autoencoder = Model(input_layer, decoded)

        # Compile the model
        autoencoder.compile(optimizer='adam', loss='mse')

        # Define the encoder model
        encoder = Model(input_layer, encoded)

        self.autoencoder = autoencoder
        self.encoder = encoder

        encoded_input = Input(shape=(n_components,))
        decoder_layers = Sequential(autoencoder.layers[-3:])
        self.decoder = Model(encoded_input, decoder_layers(encoded_input))

    def fit(self,
            data,
            epochs=100,
            batch_size=256,
            shuffle=True,
            validation_split=0.2):
        """
        Fit the autoencoder model to the data
        """
        self.autoencoder.fit(data, data,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_split=validation_split,
                             callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        return self

    def transform(self, data):
        """
        Transform the data using the encoder model
        """
        return self.encoder.predict(data)

    def inverse_transform(self, data):
        """
        Recreate the data using the autoencoder model
        """
        return self.decoder.predict(data)
