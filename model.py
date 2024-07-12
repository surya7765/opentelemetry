import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    # Create sequences
    sequence_length = 60
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

def predict(model, x_test):
    predictions = model.predict(x_test)
    return predictions

# Example usage
# data = load_data('data/stock_prices.csv')
# x, y, scaler = preprocess_data(data)
# model = create_model((x.shape[1], 1))
# model = train_model(model, x, y)
# predictions = predict(model, x_test)
