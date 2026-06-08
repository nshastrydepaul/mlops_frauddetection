# Multi-stage build for Fraud-Anomoly Detection & Behavioral Analytics

# Stage 1 - Builder
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    make \
    libffi-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    -r requirements.txt

# Stage 2 - Runtime
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /usr/local /usr/local

COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT ["python", "-m", "mlops_frauddetection.train_model"]