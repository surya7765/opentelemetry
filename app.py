import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, Path, Query, Header, Body
from pydantic import BaseModel
from model import load_data, preprocess_data, create_model, train_model, predict
from metrics import instrument_app, logger

# Define a model for the request body
class InstanceInfo(BaseModel):
    instance_id: str
    serviceApiKey: str
    
app = FastAPI()
instrument_app(app)

trained_model = None

SERVICE_ID = "1002"
ERROR_MESSAGE = 'Service API key not provided in the authorization header'
INVALID_DETAILS_MSG = "Invalid details. Please check your instance ID and service API key."
CHECK_SERVICE_API_KEY_MSG = "Checking if service API key is provided."
UPDATING_TELEMETRY_DETAILS_MSG = "Updating telemetry details."
VALUE_ERROR_MISSING = "ValueError: Missing required parameters."

@app.get("/test/123/v1/{instance_id}/fetch")
def fetch_data_from_adaptor(
    instance_id: str = Path(..., convert_underscores=False),
    serviceApiKey: str = Header(..., convert_underscores=False)
):
    """
    Fetches data from the Jira data adaptor and other details from Jira.

    Args:
        instance_id (str): The ID of the instance.
        serviceApiKey (str): The service API key for authentication.

    Returns:
        dict: The data fetched from the Jira data adaptor.
    """
    # Log messages for telemetry
    logger.info(CHECK_SERVICE_API_KEY_MSG, extra={"instanceId": instance_id})
    logger.info(UPDATING_TELEMETRY_DETAILS_MSG, extra={"instanceId": instance_id})

    # Here, you would add the logic to fetch data from the Jira data adaptor
    logger.info("Fetching data from the Jira data adaptor", extra={"instanceId": instance_id})

    # Placeholder for actual fetch logic
    data = {"message": "Data fetched successfully"}

    return data


@app.post("/test/123/v1/{instance_id}/train")
async def train(
    instance_id: str = Path(...),
    serviceApiKey: str = Header(...)):
    global trained_model
    try:
        data = load_data("data/stock_prices.csv")
        x_train, y_train = preprocess_data(data)
        model = create_model((x_train.shape[1], 1))
        trained_model = train_model(model, x_train, y_train)
        
        # Log the success message
        logger.info("Model trained successfully", extra={"instanceId": instance_id})
        return {"message": "Model trained successfully"}
    except Exception as e:
        logger.error(f"Error during training: {e}", extra={"instanceId": instance_id})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def make_prediction(
    data: List[float] = Body(...),
    instance_id: str = Query(...),
    api_key: str = Header(...)):
    try:
        if trained_model is None:
            error_message = "Model not trained yet. Please train the model first."
            logger.error(error_message, extra={"instanceId": instance_id})
            raise HTTPException(status_code=400, detail=error_message)
        
        new_data = pd.DataFrame(data, columns=["Close"])
        x_new, _ = preprocess_data(new_data)
        predictions = predict(trained_model, x_new)
        
        # Log the predictions
        logger.info(f"Predictions made successfully: {predictions.tolist()}", extra={"instanceId": instance_id})
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Error during prediction: {e}", extra={"instanceId": instance_id})
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
