FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY data/sample_customers.csv ./data/sample_customers.csv

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir .

CMD ["churn-report", "data/sample_customers.csv", "--config", "examples/config.json", "--out", "outputs/demo"]
