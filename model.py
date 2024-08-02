import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from opentelemetry import trace
from fastapi import HTTPException
import logging
import time
from metrics import record_api_call, track_latency, record_error

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
scaler = None

def load_data(file_path):
    with trace.get_tracer(__name__).start_as_current_span("load_data"):
        start_time = time.time()
        try:
            data = pd.read_csv(file_path)
            record_api_call()
            return data[:5000]
        except Exception as e:
            record_error()
            logger.error(f"Error loading data: {e}")
            raise HTTPException(status_code=500, detail="Error loading data")
        finally:
            track_latency(start_time)

def preprocess_data(data):
    with trace.get_tracer(__name__).start_as_current_span("preprocess_data"):
        start_time = time.time()
        try:
            global scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
            sequence_length = 60
            x, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                x.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            x, y = np.array(x), np.array(y)
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
            record_api_call()
            return x, y
        except Exception as e:
            record_error()
            logger.error(f"Error preprocessing data: {e}")
            raise HTTPException(status_code=500, detail="Error preprocessing data")
        finally:
            track_latency(start_time)

def create_model(input_shape):
    with trace.get_tracer(__name__).start_as_current_span("create_model"):
        start_time = time.time()
        try:
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            record_api_call()
            return model
        except Exception as e:
            record_error()
            logger.error(f"Error creating model: {e}")
            raise HTTPException(status_code=500, detail="Error creating model")
        finally:
            track_latency(start_time)

def train_model(model, x_train, y_train):
    with trace.get_tracer(__name__).start_as_current_span("train_model"):
        start_time = time.time()
        try:
            model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
            record_api_call()
            return model
        except Exception as e:
            record_error()
            logger.error(f"Error training model: {e}")
            raise HTTPException(status_code=500, detail="Error training model")
        finally:
            track_latency(start_time)

def predict(model, x_test):
    with trace.get_tracer(__name__).start_as_current_span("predict"):
        start_time = time.time()
        try:
            predictions = model.predict(x_test)
            record_api_call()
            return predictions
        except Exception as e:
            record_error()
            logger.error(f"Error predicting: {e}")
            raise HTTPException(status_code=500, detail="Error predicting")
        finally:
            track_latency(start_time)
