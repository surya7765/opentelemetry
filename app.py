from fastapi import FastAPI, Request, HTTPException
from metrics import record_metrics
from model import load_data, preprocess_data, create_model, train_model, predict
import pandas as pd

app = FastAPI()

# Load and preprocess data
data = load_data("data/stock_prices.csv")
x, y, scaler = preprocess_data(data)
input_shape = (x.shape[1], 1)

# Create and train model
model = create_model(input_shape)
model = train_model(model, x, y)

@app.post("/predict")
async def predict_endpoint(request: Request):
    try:
        # Read and parse the incoming JSON payload
        payload = await request.json()
        print("Received payload:", len(payload))  # Print payload for debugging
        
        features = payload.get("features")

        print(features)
        
        if not isinstance(features, list):
            raise HTTPException(status_code=400, detail="Features must be a list of numerical values")
        
        # if len(features) != 60:
        #     raise HTTPException(status_code=400, detail="Features must be a list of 60 numerical values")
        
        # Convert features to a DataFrame
        df = pd.DataFrame({"Close": features})
        
        # Preprocess data
        x_test, _, _ = preprocess_data(df)
        
        # Predict
        predictions = predict(model, x_test)
        result = {"predictions": predictions.tolist()}
        
        # Record metrics
        record_metrics(request, result)
        
        return result
    
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except Exception as e:
        # Log the error or print it for debugging
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@app.get("/metrics")
async def metrics_endpoint():
    # Return a dummy response as metrics will be collected and exported
    return {"status": "metrics collected"}
