FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# LightGBM requires the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy project metadata and source
COPY pyproject.toml .
COPY src/ ./src/

# Install the project and its dependencies
RUN pip install --no-cache-dir .

EXPOSE 8000

# Run the API using uvicorn
CMD ["sh", "-c", "python -m uvicorn bot_or_not.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]