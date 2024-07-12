from flask import Flask, request, jsonify
import numpy as np
from model import load_data, preprocess_data, create_model, train_model, predict
from metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT, API_CALLS, 
    UNIQUE_USERS, PEAK_USAGE, ENDPOINT_POPULARITY, 
    PREDICTION_ACCURACY, MODEL_VERSION
)
from opentelemetry import trace
from opentelemetry.exporter.zipkin.proto.http import ZipkinExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

app = Flask(__name__)

# Initialize OpenTelemetry tracing
resource = Resource(attributes={"service.name": "ml-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

zipkin_exporter = ZipkinExporter(
    endpoint="http://localhost:9411/api/v2/spans"
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(zipkin_exporter)
)

FlaskInstrumentor().instrument_app(app)

# Load and preprocess data
data = load_data('data/stock_prices.csv')
x, y, scaler = preprocess_data(data)
input_shape = (x.shape[1], 1)
model = create_model(input_shape)

@app.route("/train", methods=["POST"])
def train():
    REQUEST_COUNT.inc()
    with tracer.start_as_current_span("train_request"):
        with REQUEST_LATENCY.time():
            try:
                # Split data into training and testing sets
                # Placeholder for actual train/test split
                x_train, y_train = x, y

                # Train model
                global model
                model = train_model(model, x_train, y_train)
                MODEL_VERSION.set(1.0)  # Example version
                return jsonify({"message": "Model trained successfully"})
            except Exception as e:
                ERROR_COUNT.inc()
                return jsonify({"error": str(e)}), 502

@app.route("/predict", methods=["POST"])
def make_prediction():
    REQUEST_COUNT.inc()
    with tracer.start_as_current_span("predict_request"):
        with REQUEST_LATENCY.time():
            try:
                # Simulate prediction logic
                data = request.json['data']
                data = np.array(data).reshape(-1, 1)
                scaled_data = scaler.transform(data)
                x_test = []
                x_test.append(scaled_data)
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
                predictions = predict(model, x_test)
                accuracy = 0.95  # Placeholder for actual accuracy calculation
                PREDICTION_ACCURACY.set(accuracy)
                return jsonify({"predictions": predictions.tolist(), "accuracy": accuracy})
            except Exception as e:
                ERROR_COUNT.inc()
                return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
