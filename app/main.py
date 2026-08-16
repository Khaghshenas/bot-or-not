from time import perf_counter

from fastapi import FastAPI, HTTPException, Response

from app.prediction import predict_traffic
from app.schemas import LogLine, PredictionResponse

from app.config import APP_NAME, APP_VERSION, APP_DESCRIPTION

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)


@app.get("/health")
def health_check() -> dict:
    """Confirm that the API is running."""

    return {"status": "healthy"}


@app.post("/predict")
def predict(log_line: LogLine, response: Response) -> PredictionResponse:
    """Predict whether a single log line represents NHT or human traffic (HT)."""

    start_time = perf_counter()
    
    try:
        result = predict_traffic(log_line)
    except Exception as error:
        # Return a controlled response without exposing internal details.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from error

    prediction_time_ms = (perf_counter() - start_time) * 1000

    # Expose the prediction latency as a response header.
    # This provides a simple technical performance measurement.
    response.headers["X-Prediction-Time-Ms"] = (f"{prediction_time_ms:.3f}")
    response.headers["X-Model-Version"] = APP_VERSION 

    return PredictionResponse(**result)