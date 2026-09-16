import logging
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, Response

from bot_or_not.api.schemas import LogLine, PredictionResponse
from bot_or_not.config import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
)
from bot_or_not.inference import predict_traffic


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
async def add_request_timing(
    request: Request,
    call_next,
) -> Response:
    start = perf_counter()

    response = await call_next(request)

    response.headers["X-Request-Time-Ms"] = (
        f"{(perf_counter() - start) * 1000:.3f}"
    )

    return response


@app.get("/health")
def health_check() -> dict[str, str]:
    """Confirm that the API process is running."""

    return {"status": "healthy"}


@app.post("/predict")
def predict(
    log_line: LogLine,
    response: Response,
) -> PredictionResponse:
    """Predict whether a log line represents NHT or HT."""

    start = perf_counter()

    try:
        result = predict_traffic(log_line.model_dump())
    except Exception as error:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from error

    response.headers["X-Model-Time-Ms"] = (
        f"{(perf_counter() - start) * 1000:.3f}"
    )
    response.headers["X-App-Version"] = APP_VERSION

    return PredictionResponse(**result)