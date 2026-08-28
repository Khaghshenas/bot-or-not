from statistics import median
from time import perf_counter

import joblib
import pandas as pd

from app.config import MODEL_FEATURES, MODEL_PATH
from app.prediction import pipeline


NUMBER_OF_RUNS = 200
WARMUP_RUNS = 10

SAMPLE = {
    "country_by_ip_address": "IT",
    "region_by_ip_address": "LI",
    "visitor_recognition_type": "ANONYMOUS",
}


def create_input_data():
    return pd.DataFrame([SAMPLE])[MODEL_FEATURES]


def benchmark_model_loading(input_data):
    """Load the model for every prediction."""

    for _ in range(WARMUP_RUNS):
        model = joblib.load(MODEL_PATH)
        model.predict_proba(input_data)

    latencies = []

    for _ in range(NUMBER_OF_RUNS):
        start = perf_counter()

        model = joblib.load(MODEL_PATH)
        model.predict_proba(input_data)

        latency_ms = (perf_counter() - start) * 1000

        latencies.append(latency_ms)

    return latencies


def benchmark_reused_model(input_data):
    """Reuse the model loaded when the application starts."""

    for _ in range(WARMUP_RUNS):
        pipeline.predict_proba(input_data)

    latencies = []

    for _ in range(NUMBER_OF_RUNS):
        start = perf_counter()

        pipeline.predict_proba(input_data)

        latency_ms = (perf_counter() - start) * 1000

        latencies.append(latency_ms)

    return latencies


def calculate_results(latencies):
    latency_series = pd.Series(latencies)

    return {
        "median_ms": median(latencies),
        "p95_ms": latency_series.quantile(0.95),
    }

def main():
    input_data = create_input_data()

    baseline_latencies = benchmark_model_loading(input_data)
    optimized_latencies = benchmark_reused_model(input_data)

    baseline = calculate_results(baseline_latencies)
    optimized = calculate_results(optimized_latencies)

    median_improvement = ((baseline["median_ms"] - optimized["median_ms"]) / baseline["median_ms"] * 100)

    p95_improvement = ((baseline["p95_ms"] - optimized["p95_ms"]) / baseline["p95_ms"] * 100)

    speedup = (baseline["median_ms"] / optimized["median_ms"])

    print("Baseline: load model per prediction")
    print(f"Median latency: "f"{baseline['median_ms']:.3f} ms")
    print(f"P95 latency: "f"{baseline['p95_ms']:.3f} ms")

    print("\nOptimized: reuse loaded model")
    print(f"Median latency: "f"{optimized['median_ms']:.3f} ms")
    print(f"P95 latency: "f"{optimized['p95_ms']:.3f} ms")

    print("\nImprovement")
    print(f"Median latency improvement: "f"{median_improvement:.1f}%")
    print(f"P95 latency improvement: "f"{p95_improvement:.1f}%")
    print(f"Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    main()