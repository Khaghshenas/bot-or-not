# Bot or Not API

A machine learning application that classifies a web traffic log line as either human traffic (`HT`) or non-human traffic (`NHT`). The service packages a trained LightGBM pipeline, its decision threshold, and its runtime configuration with a FastAPI application.

> The source dataset cannot be published. The repository therefore includes the model-development notebooks and the deployable model artifact, but not the raw data.


## What this project demonstrates

- Session-aware train, validation, and test splitting to reduce leakage between requests from the same session.
- A scikit-learn pipeline that combines one-hot encoding with a LightGBM classifier.
- Threshold selection based on explicit false-positive and false-negative costs.
- A versioned model artifact containing the fitted pipeline, selected threshold, version, and metadata.
- A FastAPI inference service with validation, controlled error responses, timing headers, and interactive OpenAPI documentation.
- Python packaging with a `src/` layout and runtime resources included through `pyproject.toml`.
- Automated API, model-artifact, and threshold-evaluation tests.
- Docker packaging and a benchmark comparing per-request model loading with model reuse.

## Target definition and assumptions

The source dataset contains multiple `ua_agent_class` values. They are mapped to the binary target as follows:

- `Browser`, `Browser Webview`, and `Mobile App` -> `HT` (`0`)
- All other classes -> `NHT` (`1`)

This mapping assumes that `Mobile App` represents human traffic and `Special` represents non-human traffic. Those assumptions should be confirmed with the data owner before using the model for business decisions.


## Model

The deployed classifier uses three categorical features:

- `country_by_ip_address`
- `region_by_ip_address`
- `visitor_recognition_type`

Requests are split by `session_id`, keeping a session entirely within one data split. Preprocessing and classification are stored in a single pipeline so training and inference apply the same transformations. Unknown categories are ignored by the one-hot encoder rather than causing inference failures.

The packaged artifact is version `v001` and contains:

- the fitted preprocessing and LightGBM pipeline;
- the selected NHT decision threshold;
- artifact version and training metadata.

The current threshold is `0.90`. A request is classified as `NHT` when its estimated NHT probability is at least this threshold; otherwise it is classified as `HT`.

### Recorded model comparison

| Model | Accuracy | Macro F1 | ROC-AUC | PR-AUC |
|:---|---:|---:|---:|---:|
| KNN | 0.982 | 0.980 | 0.984 | 0.974 |
| LightGBM | 0.982 | 0.981 | 0.994 | 0.985 |

LightGBM was selected because it produced stronger probability-ranking metrics and offers better inference scaling than KNN. See the notebooks for the complete development and analysis workflow:

- [`01_data_preparation_and_lightgbm_training.ipynb`](notebooks/01_data_preparation_and_lightgbm_training.ipynb)
- [`02_error_analysis_and_ablation_experiments.ipynb`](notebooks/02_error_analysis_and_ablation_experiments.ipynb)

## Project structure

```text
.
├── benchmarks/
│   └── benchmark_model_loading.py
├── notebooks/
│   ├── 01_data_preparation_and_lightgbm_training.ipynb
│   └── 02_error_analysis_and_ablation_experiments.ipynb
├── src/bot_or_not/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── resources/
│   │   ├── bot_detection_pipeline_v001.joblib
│   │   └── config.yaml
│   ├── artifacts.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── inference.py
│   └── training.py
├── tests/
├── Dockerfile
└── pyproject.toml
```

`pyproject.toml` defines the installable package and its dependencies. The YAML configuration and serialized model are package data under `src/bot_or_not/resources/`, so they are available after installation instead of depending on the original repository layout.


## API

### Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Confirms that the API process is running. |
| `POST` | `/predict` | Classifies one web-traffic log line. |
| `GET` | `/docs` | Opens the interactive Swagger UI. |

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

`country_by_ip_address`, `region_by_ip_address`, and `referrer_without_parameters` may be `null` or omitted. The other fields are required by the API schema. Only the three model features listed above are passed to the classifier.

### Prediction response

```json
{
  "prediction": "HT",
  "probability_nht": 0.00868583080920436
}
```

Successful prediction responses also include:

- `X-Model-Time-Ms`: feature preparation and model-inference time;
- `X-Request-Time-Ms`: total server-side request time;
- `X-App-Version`: API version from the packaged configuration.

Invalid request bodies return FastAPI's `422` validation response. Unexpected prediction errors are logged by the server and returned to clients as a controlled `500` response without exposing internal exception details.


## Local development

### Requirements

- Python 3.11 or later
- Git

Clone the repository and enter it:

```bash
git clone https://github.com/Khaghshenas/bot-or-not.git
cd bot-or-not
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package in editable mode with its development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Start the API:

```bash
python -m uvicorn bot_or_not.api.main:app --host 0.0.0.0 --port 8000
```

Open:

- Health check: <http://localhost:8000/health>
- Swagger UI: <http://localhost:8000/docs>

## Tests

Run the complete test suite from the repository root:

```bash
python -m pytest -v
```

The tests cover:

- health and prediction endpoints;
- request validation and controlled error responses;
- response structure and observability headers;
- deterministic and concurrent predictions;
- artifact structure, threshold validity, and model probabilities;
- threshold metrics and invalid evaluation inputs.

## Model-loading benchmark

The API imports and validates the model artifact once when the inference module starts, then reuses the loaded pipeline for every request. The benchmark compares that approach with loading and validating the artifact before every prediction.

Run it from the repository root:

```bash
python -m benchmarks.benchmark_model_loading
```

The benchmark performs 10 warm-up runs and 200 measured predictions for each approach, then reports median and p95 latency. It measures the Python model-prediction path rather than complete HTTP latency, so results depend on the operating system, hardware, Python environment, and background load.

## Docker

Build the image from the repository root:

```bash
docker build -t bot-or-not-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 bot-or-not-api
```

The image installs the project from `pyproject.toml`, including the packaged configuration and model artifact. The container listens on port `8000` by default. A different internal port can be supplied with `PORT`:

```bash
docker run --rm -e PORT=8080 -p 8080:8080 bot-or-not-api
```

## Reproducibility notes

- Training uses a fixed random seed of `42`.
- Splits are stratified and grouped by `session_id`.
- The deployed artifact stores the preprocessing pipeline and classifier together.
- The selected decision threshold and artifact metadata are stored beside the fitted pipeline.
- Application configuration and the model artifact are included as package resources.
- Runtime and development dependencies are declared in `pyproject.toml`.

Because the raw dataset is private, the training notebooks cannot be rerun from this repository alone. The included artifact allows the API and inference tests to run without the training data.

## Live demo

A demonstration deployment is available on Render:

- Health check: <https://bot-or-not-fn7i.onrender.com/health>
- Swagger UI: <https://bot-or-not-fn7i.onrender.com/docs>

The service uses Render's free tier and may take approximately one minute to start after a period of inactivity.
