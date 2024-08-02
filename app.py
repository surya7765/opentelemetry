from fastapi import FastAPI, HTTPException
import uvicorn
from metrics import collect_metrics
from model import load_data, preprocess_data, create_model, train_model, predict

app = FastAPI()

model = None
x_train, y_train = None, None

@app.post("/train")
def train():
    global model, x_train, y_train
    data = load_data("data/stock_prices.csv")
    x_train, y_train = preprocess_data(data)
    model = create_model((x_train.shape[1], 1))
    trained_model = train_model(model, x_train, y_train)
    return {"message": "Model trained successfully"}

@app.post("/predict")
def make_prediction():
    if model is None:
        raise HTTPException(status_code=400, detail="Model is not trained yet")
    predictions = predict(model, x_train)
    return {"predictions": predictions.tolist()}

@app.get("/metrics")
def get_metrics():
    metrics_data = collect_metrics()
    return {"Event": "/fetch", "Metrics": metrics_data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
