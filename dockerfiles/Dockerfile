
# Multi-stage build for Fraud-Anomoly Detection & Behavioral Analytics

# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    gcc \
    g++ \
    curl \
    git \
    libffi-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --target=/root/.local/lib/python3.11/site-packages

# Stage 2: Runtime
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install runtime libraries
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy project code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT ["python", "-m", "mlops_frauddetection.train_model"]
