import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self):
        self.model = self._build_model()
        self._train()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def _train(self):
        X = np.random.uniform(40, 90, size=(100, 10))
        y = X.mean(axis=1)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.model.fit(X, y, epochs=5, verbose=0)

    def predict(self, cpu_data):
        X = np.array(cpu_data).reshape((1, 10, 1))
        return float(self.model.predict(X)[0][0])
