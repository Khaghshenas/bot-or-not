from pydantic import BaseModel, ConfigDict, Field


class LogLine(BaseModel):
    """A single web-request log line."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "epoch_ms": 1520280002397,
                "session_id": "ee391655f5680a7bfae0019450aed396",
                "country_by_ip_address": "IT",
                "region_by_ip_address": "LI",
                "url_without_parameters": (
                    "https://www.bol.com/nl/p/example-product/12345/"
                ),
                "referrer_without_parameters": (
                    "https://www.bol.com/nl/"
                ),
                "visitor_recognition_type": "ANONYMOUS",
            }
        }
    )

    epoch_ms: int = Field(
        ...,
        description="Milliseconds since the Unix epoch",
    )
    session_id: str = Field(
        ...,
        description="Visitor-session identifier",
    )
    url_without_parameters: str = Field(
        ...,
        description="Requested URL without query parameters",
    )
    visitor_recognition_type: str = Field(
        ...,
        description="Visitor recognition type",
    )

    country_by_ip_address: str | None = Field(
        default=None,
        description="Country inferred from IP address",
    )
    region_by_ip_address: str | None = Field(
        default=None,
        description="Region inferred from IP address",
    )
    referrer_without_parameters: str | None = Field(
        default=None,
        description="Referrer URL without query parameters",
    )


class PredictionResponse(BaseModel):
    """Prediction returned by the bot-detection model."""

    prediction: str = Field(
        ...,
        description="Predicted traffic type: NHT or HT",
    )
    probability_nht: float = Field(
        ...,
        ge=0,
        le=1,
        description="Estimated probability of NHT",
    )