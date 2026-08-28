import logging
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, Response

from app.config import APP_DESCRIPTION, APP_NAME, APP_VERSION
from app.prediction import predict_traffic
from app.schemas import LogLine, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    response.headers["X-Request-Time-Ms"] = (
        f"{(perf_counter() - start) * 1000:.3f}"
    )
    return response

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
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from error

    prediction_time_ms = (perf_counter() - start_time) * 1000

    # Expose the prediction latency as a response header.
    response.headers["X-Model-Time-Ms"] = f"{prediction_time_ms:.3f}"
    response.headers["X-App-Version"] = APP_VERSION 

    return PredictionResponse(**result)