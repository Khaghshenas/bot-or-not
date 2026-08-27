# Bot or Not API

A machine learning application that classifies a web traffic log line as either human traffic (`HT`) or non-human traffic (`NHT`). 

Because the dataset cannot be shared, this project focuses mainly on deployment. The model-development process—including data preprocessing, training, evaluation, and model selection—is completed in a Jupyter notebook. The final LightGBM classifier is exported as bot_detection_pipeline.joblib.

The deployment code included in this project loads the saved model, processes incoming web traffic log lines, generates predictions, and returns the corresponding HT or NHT classification through the application interface.

## Assumptions

The supplied dataset contains several user-agent classes (ua_agent_class). For the binary API model, the classes are mapped as follows:

- `Browser`, `Browser Webview`, and `Mobile App` -> `HT`
- All remaining classes -> `NHT`

I assumed `Mobile App` is HT and `Special` is NHT. This should be confirmed with the data owner because the business definition of NHT is not included in the dataset.

## Solution overview

The solution consists of:

- A Jupyter notebook for model development and evaluation.
- A scikit-learn pipeline containing one-hot encoding and a LightGBM classifier.
- A FastAPI application with health and prediction endpoints.
- Swagger UI for interactive API documentation.
- API and model tests implemented with `pytest` and FastAPI's `TestClient`.
- A benchmark measuring the model-loading optimization.
- Docker packaging for reproducible local execution.


## Model summary

The deployed model is a binary LightGBM classifier using:

- `country_by_ip_address`
- `region_by_ip_address`
- `visitor_recognition_type`

Requests are split by `session_id` to prevent the same session from appearing in both training and test data. Preprocessing and classification are stored in one pipeline to keep training and inference consistent.

| Model | Accuracy | Macro F1 | ROC-AUC | PR-AUC |
|:---|---:|---:|---:|---:|
| KNN | 0.982 | 0.980 | 0.984 | 0.974 |
| LightGBM | 0.982 | 0.981 | 0.994 | 0.985 |

LightGBM was selected for deployment because it provides stronger probability ranking and better inference scaling than KNN.

See [`notebooks/bot_or_not_lightgbm.ipynb`](notebooks/bot_or_not_lightgbm.ipynb) for data exploration, target definition, model training, and evaluation.

## API

### Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Confirms that the service is running. |
| `POST` | `/predict` | Returns the predicted traffic type and NHT probability for one log line. |
| `GET` | `/docs` | Interactive Swagger UI. |

### Prediction request

```json
{
  "epoch_ms": 1520280002397,
  "session_id": "ee391655f5680a7bfae0019450aed396",
  "country_by_ip_address": "IT",
  "region_by_ip_address": "LI",
  "url_without_parameters": "https://www.bol.com/nl/p/example-product/12345/",
  "referrer_without_parameters": "https://www.bol.com/nl/",
  "visitor_recognition_type": "ANONYMOUS"
}
```

### Prediction response

```json
{
  "prediction": "HT",
  "probability_nht": 0.00868583080920436
}
```

## Run locally

### Prerequisites

- Python 3.11 or later

### Installation

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Start the API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open the interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

## Tests

Run the automated tests from the project root:

```bash
python -m pytest -v
```

The current test suite checks:

- Health and prediction endpoints.
- Response schema and observability headers.
- Missing, invalid, and unknown input values.
- Controlled internal-error responses.
- Deterministic predictions and concurrent requests.
- Model artifact loading and prediction consistency.

## API performance optimization

The serialized **model pipeline is loaded once when the application starts** rather than once per request. This removes repeated disk I/O and deserialization from the prediction path.

### Benchmark methodology

I compared two inference approaches:

- **Baseline:** load and deserialize the model before every prediction.
- **Optimized:** load the model once at application startup and reuse it for every prediction.

The benchmark used:

- 10 warm-up runs.
- 200 measured predictions.
- One input record per prediction.
- Median and p95 latency.
- Local execution on Windows.

The benchmark measures the model prediction path, not complete HTTP request latency. Results may vary depending on hardware and system load.

Run the benchmark from the project root:

```bash
python -m benchmarks.benchmark_model_loading
```

### Benchmark results

| Metric | Load model per prediction | Reuse loaded model |
|:---|---:|---:|
| Median latency | 19.03 ms | 6.83 ms |
| P95 latency | 24.65 ms | 7.91 ms |

In this run, reusing the loaded model reduced median latency by **64.1%**, reduced p95 latency by **67.9%**, and provided a **2.8x speedup**.

Across repeated local runs, the median latency reduction was approximately **57–64%**, with a **2.3–2.8x speedup**. This variation is expected in a local benchmark because of operating-system scheduling and background processes.

## Technical observability

Successful responses expose:

- `X-Model-Time-Ms`: feature preparation and model inference latency.
- `X-Request-Time-Ms`: total server-side request latency.
- `X-App-Version`: application version.

The distinction helps determine whether latency comes from model inference or API overhead. 

## Docker

The API can be built and run locally with Docker.

### Prerequisite

Install and start Docker Desktop.

### Build the image

From the project root, run:

```bash
docker build -t bot-or-not-api .
```

### Run the container

```bash
docker run --rm -p 8000:8000 bot-or-not-api
```

Port `8000` on the local machine is mapped to port `8000` inside the container.

Verify the service at:

- Health endpoint: [http://localhost:8000/health](http://localhost:8000/health)
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

## Reproducibility

The model-training workflow is documented in `notebooks/bot_or_not_lightgbm.ipynb`. The notebook uses a fixed random seed and saves the complete preprocessing and prediction pipeline to `models/bot_detection_pipeline.joblib`.

## Live demo

A demonstration deployment is available at:

- Health check: [https://bot-or-not-fn7i.onrender.com/health](https://bot-or-not-fn7i.onrender.com/health)
- Swagger UI: [https://bot-or-not-fn7i.onrender.com/docs](https://bot-or-not-fn7i.onrender.com/docs)

The service uses Render's free tier and may require approximately one minute to start after a period of inactivity.