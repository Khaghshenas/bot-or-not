# Bot or Not API

A machine learning application that classifies a web traffic log line as either human traffic (`HT`) or non-human traffic (`NHT`). The service packages a trained LightGBM pipeline, its decision threshold, and its runtime configuration with a FastAPI application.

> The source dataset cannot be published. The repository therefore includes the model-development notebooks and the deployable model artifact, but not the raw data.


## What this project demonstrates

- Session-aware train, validation, and test splitting to reduce leakage between requests from the same session.
- A scikit-learn pipeline that combines one-hot encoding with a LightGBM classifier.
- Threshold selection based on explicit false-positive and false-negative costs.
- Command-line training, with dataset validation, session-leakage checks, validation-based threshold selection, and held-out
  test evaluation.
- Training-run metadata, dataset fingerprinting, artifact reload verification, and explicit completion markers.
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
│   ├── train.py
│   └── training.py
├── tests/
├── Dockerfile
├── Dockerfile.training
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

## Training

Training can run independently of Jupyter through the `bot_or_not.train` module. Notebooks remain available for exploration,
visualization, and error analysis.

### Responsibilities

- `training.py` defines and fits the preprocessing/classifier pipeline.
- `train.py` coordinates data loading, validation, splitting, training,
  threshold selection, evaluation, and output creation.
- The FastAPI application loads a previously saved artifact for inference.
  Starting the API does not initiate training.

### Prerequisites

The source dataset is private and is not included in this repository.
Supply an authorized local CSV snapshot.

The CSV must contain:

| Column | Requirement |
|:---|:---|
| `country_by_ip_address` | String values or missing values |
| `region_by_ip_address` | String values or missing values |
| `visitor_recognition_type` | String values or missing values |
| `session_id` | Non-null, non-empty session identifiers |
| `ua_agent_class` | Non-null, non-empty source labels |

All columns must exist, even when missing feature values are permitted. Feature columns are controlled by `model.features` in the packaged `config.yaml`; update these requirements if that configuration changes.

CSV parsing uses pandas' default type and missing-value inference. A feature column parsed as numeric rather than string will fail validation.

The training job maps `Browser`, `Browser Webview`, and `Mobile App` to human traffic (`0`), and all other non-null source labels to non-human traffic (`1`). This mapping must be confirmed with the data owner. Unknown source-label values are not currently rejected.

Provide enough sessions and class examples for the grouped split to produce both target classes in each of the training, validation, and test sets. The job rejects splits that do not contain both classes.

### Start a training run

From the repository root:

```bash
python -m bot_or_not.train \
  --data-path data/clickdata.csv \
  --output-dir runs/run-001 \
  --model-version v002
```

Show command-line help:

```bash
python -m bot_or_not.train --help
```

| Argument | Meaning |
|:---|:---|
| `--data-path` | Path to a local training CSV |
| `--output-dir` | New directory for this run's outputs |
| `--model-version` | Non-empty candidate version stored in the artifact |

Relative paths are resolved from the current working directory.
The output directory must not already exist. Use a fresh directory for
each run or retry.

The version argument labels the artifact; it does not enforce uniqueness
across different output directories.

### Training workflow

The job:

1. Calculates the input file's SHA-256 fingerprint.
2. Validates required columns, session identifiers, labels, and feature types.
3. Constructs the binary target using the existing source-label mapping.
4. Creates approximately 60% training, 20% validation, and 20% test splits,
   keeping sessions separate.
5. Checks that every split contains both target classes and that no session
   appears in multiple splits.
6. Fits preprocessing and LightGBM using the training split only.
7. Evaluates thresholds from `0.05` through `0.95`, in increments of `0.05`,
   using validation labels and validation probabilities.
8. Selects the threshold with the lowest configured validation business cost.
9. Evaluates the fixed candidate on the held-out test split.
10. Saves the artifact, reloads it, and verifies that its predictions,
    threshold, and version are preserved.
11. Saves evaluation reports and writes a completion marker.

Split proportions are approximate because sessions remain intact.

The random seed, model features, and false-positive/false-negative costs come from the packaged configuration. Model hyperparameters are defined in `training.py`.

Test data is not used to fit preprocessing, fit the classifier, or select the threshold. Avoid repeatedly tuning the model against reported test results; maintain a suitable held-out evaluation policy.

### Output files

A successful run produces:

```text
runs/run-001/
├── model.joblib
├── metrics.json
├── validation_thresholds.csv
└── SUCCESS
```

| File | Contents |
|:---|:---|
| `model.joblib` | Fitted preprocessing/classifier pipeline, threshold, version, and metadata |
| `metrics.json` | Run metadata, test metrics, completion timestamp, and reload-verification status |
| `validation_thresholds.csv` | Validation metrics and business cost for each candidate threshold |
| `SUCCESS` | Completion marker written after artifact verification and report creation |

Metadata records the dataset SHA-256, model version, start time, random seed, feature names, split strategy and sizes configured error costs, and installed pandas, scikit-learn, LightGBM, and joblib versions.

The test report includes the selected threshold, confusion-matrix counts, precision, recall, false-positive rate, specificity, F1, predicted-positive rate, and business cost.

### Logs and failure handling

The job logs major stages and the final test metrics to the process log.

An exception is logged and re-raised, giving the process a nonzero exit status. Failures after output-directory creation may leave partial files.

Downstream consumers must require the `SUCCESS` marker before using a run's outputs. Use a new output directory when retrying a failed run.

`SUCCESS` confirms execution and artifact verification. It does not indicate that the candidate meets business acceptance criteria.

### Deployment is separate

Training does not replace the model currently loaded by the API and does not deploy automatically.

The current API loads its model from packaged resources, as configured in `src/bot_or_not/resources/config.yaml`.

To release an approved candidate with this design:

1. Review evaluation results against explicit release criteria.
2. Copy the approved `model.joblib` into
   `src/bot_or_not/resources/` using an appropriate versioned filename.
3. Update `paths.model` in the packaged configuration.
4. Run the tests against the updated artifact.
5. Rebuild and deploy the inference image.
6. Retain the previous image for rollback.

Use compatible dependency versions for training and inference. Recording versions helps trace a run; it does not itself lock the environment.

### Training with Docker

Build the training image from the repository root:

```bash
docker build \
  -f Dockerfile.training \
  -t bot-or-not-training .
```

Create a host output directory and start a training container:

```bash
mkdir -p runs

docker run --rm \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/runs:/runs" \
  bot-or-not-training \
  --data-path /data/clickdata.csv \
  --output-dir /runs/run-001 \
  --model-version v002
```

These mount examples use Bash syntax. Input data is mounted read-only, and the output mount preserves results after the container exits.

`Dockerfile.training` starts the training module instead of Uvicorn. The container exits when training finishes. 

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
