from pathlib import Path


# Application metadata
APP_NAME = "Bot or Not API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "API for classifying web traffic as NHT or human traffic (HT)."


# File paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "bot_detection_pipeline.joblib"


# Input columns expected by the trained pipeline
# These are the features used during model training.
MODEL_FEATURES = [
    "country_by_ip_address",
    "region_by_ip_address",
    "visitor_recognition_type",
]


# The binary model was trained with 1 representing NHT
NHT_CLASS_LABEL = 1