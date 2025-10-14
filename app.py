from flask import Flask, request, jsonify
from metrics import update_system_metrics
from model import load_data, preprocess_data, create_model, train_model, predict

app = Flask(__name__)

model = None
x_train, y_train = None, None

@app.route("/train", methods=["POST"])
def train():
    global model, x_train, y_train
    data = load_data("data/stock_prices.csv")
    x_train, y_train = preprocess_data(data)
    model = create_model((x_train.shape[1], 1))
    trained_model = train_model(model, x_train, y_train)
    return jsonify({"message": "Model trained successfully"})

@app.route("/predict", methods=["POST"])
def make_prediction():
    if model is None:
        return jsonify({"detail": "Model is not trained yet"}), 400
    predictions = predict(model, x_train)
    return jsonify({"predictions": predictions.tolist()})

@app.route("/metrics", methods=["GET"])
def get_metrics():
    metrics_data = update_system_metrics()
    return jsonify({"Event": "/fetch", "Metrics": metrics_data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
