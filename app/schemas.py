from pydantic import BaseModel, ConfigDict, Field

# The current model only uses country, region, and visitor recognition type.
# However, the API accepts a complete unlabeled log line from the dataset.
# This keeps the input contract compatible with the source data and allows
# additional features to be introduced later without changing the API.

class LogLine(BaseModel):
    """A single web request in the same format as the original dataset."""


    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "epoch_ms": 1520280002397,
                "session_id": "ee391655f5680a7bfae0019450aed396",
                "country_by_ip_address": "IT",
                "region_by_ip_address": "LI",
                "url_without_parameters": ("https://www.bol.com/nl/p/example-product/12345/"),
                "referrer_without_parameters": ("https://www.bol.com/nl/"),
                "visitor_recognition_type": "ANONYMOUS",
            }
        }
    )
    
    epoch_ms: int = Field(..., description="Request timestamp in milliseconds since Unix epoch")

    session_id: str = Field(..., description="Identifier of the visitor session")

    country_by_ip_address: str | None = Field(
        default=None,
        description="Country inferred from the IP address",
    )
    region_by_ip_address: str | None = Field(
        default=None,
        description="Region inferred from the IP address",
    )
    url_without_parameters: str | None = Field(
        default=None,
        description="Requested URL without query parameters",
    )
    referrer_without_parameters: str | None = Field(
        default=None,
        description="Referrer URL without query parameters",
    )
    visitor_recognition_type: str | None = Field(
        default=None,
        description="Visitor recognition type, such as ANONYMOUS or RECOGNIZED or LOGGEDIN",
    )


class PredictionResponse(BaseModel):
    """Prediction returned by the bot-detection model."""

    prediction: str = Field(..., description="Predicted traffic type: NHT or HT")

    probability_nht: float = Field(
        ...,
        ge=0,
        le=1,
        description="Estimated probability that the request is NHT",
    )