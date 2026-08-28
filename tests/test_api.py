from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app.main import app

from app.config import APP_VERSION

client = TestClient(app)

SAMPLE = {
    "epoch_ms": 1520280002397,
    "session_id": "ee391655f5680a7bfae0019450aed396",
    "country_by_ip_address": "IT",
    "region_by_ip_address": "LI",
    "url_without_parameters": "https://www.bol.com/nl/p/example-product/12345/",
    "referrer_without_parameters": "https://www.bol.com/nl/",
    "visitor_recognition_type": "ANONYMOUS",
}


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_contract():
    response = client.post("/predict", json=SAMPLE)

    assert response.status_code == 200

    body = response.json()

    assert set(body) == {"prediction", "probability_nht"}
    assert body["prediction"] in {"HT", "NHT"}
    assert 0 <= body["probability_nht"] <= 1

    assert float(response.headers["X-Model-Time-Ms"]) >= 0
    assert float(response.headers["X-Request-Time-Ms"]) >= 0
    assert float(response.headers["X-Request-Time-Ms"]) >= float(response.headers["X-Model-Time-Ms"])
    assert response.headers["X-App-Version"] == APP_VERSION


def test_prediction_is_deterministic():
    first = client.post("/predict", json=SAMPLE)
    second = client.post("/predict", json=SAMPLE)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()


def test_missing_optional_values_are_accepted():
    sample = {
        "epoch_ms": SAMPLE["epoch_ms"],
        "session_id": SAMPLE["session_id"],
        "url_without_parameters": SAMPLE["url_without_parameters"],
        "visitor_recognition_type": SAMPLE["visitor_recognition_type"]
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    assert response.json()["prediction"] in {"HT", "NHT"}


def test_unknown_categories_are_accepted():
    sample = {
        "epoch_ms": SAMPLE["epoch_ms"],
        "session_id": SAMPLE["session_id"],
        "url_without_parameters": SAMPLE["url_without_parameters"],
        "referrer_without_parameters": SAMPLE["referrer_without_parameters"],
        "country_by_ip_address": "UNSEEN_COUNTRY",
        "region_by_ip_address": "UNSEEN_REGION",
        "visitor_recognition_type": "UNSEEN_TYPE",
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200


def test_missing_required_fields_return_422():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_invalid_field_type_returns_422():
    sample = {
        "epoch_ms": "not-a-timestamp",
        "session_id": SAMPLE["session_id"],
        "url_without_parameters": SAMPLE["url_without_parameters"],
        "referrer_without_parameters": SAMPLE["referrer_without_parameters"],
        "country_by_ip_address": "UNSEEN_COUNTRY",
        "region_by_ip_address": "UNSEEN_REGION",
        "visitor_recognition_type": "UNSEEN_TYPE",
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 422


def test_internal_error_returns_controlled_500(monkeypatch):
    def failing_prediction(_):
        raise RuntimeError("Sensitive internal information")

    monkeypatch.setattr("app.main.predict_traffic", failing_prediction)

    response = client.post("/predict", json=SAMPLE)

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    assert "Sensitive internal information" not in response.text


def test_multiple_requests():
    def send_request(_):
        return client.post("/predict", json=SAMPLE)

    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(send_request, range(8)))

    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["prediction"] in {"HT", "NHT"} for response in responses)