from pydantic import BaseModel, ConfigDict, Field

# The current model uses country, region, and visitor recognition type.
# The API accepts the complete unlabeled log-line schema to remain compatible
# with the source data and support future features without changing the contract.

class LogLine(BaseModel):
    """A single web request in the same format as the original dataset."""


    # Adds an example request payload to the generated documentation.
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


    # Required because these fields are always present in the source data.
    # Visitor recognition type is also used by the deployed model. Allowing 
    # it to be missing would reduce prediction reliability.
    epoch_ms: int = Field(..., description="Request timestamp in milliseconds since Unix epoch")
    session_id: str = Field(..., description="Identifier of the visitor session")
    url_without_parameters: str = Field(..., description="Requested URL without query parameters")
    visitor_recognition_type: str = Field(..., description="Visitor recognition type")

    # Optional because these fields contain missing values in the source data.
    country_by_ip_address: str | None = Field(default=None, description="Country inferred from the IP address")
    region_by_ip_address: str | None = Field(default=None, description="Region inferred from the IP address")
    referrer_without_parameters: str | None = Field(default=None, description="Referrer URL without query parameters")


class PredictionResponse(BaseModel):
    """Prediction returned by the bot-detection model."""

    prediction: str = Field(..., description="Predicted traffic type: NHT or HT")

    probability_nht: float = Field(
        ...,
        ge=0,
        le=1,
        description="Estimated probability that the request is NHT",
    )