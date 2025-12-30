FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY src ./src
COPY artifacts ./artifacts
COPY knowledge ./knowledge
COPY ui ./ui
COPY run.py .

CMD ["python", "run.py"]