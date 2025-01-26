FROM python:3.12-slim as builder

WORKDIR /app

# Install poetry
RUN pip install poetry
RUN poetry config virtualenvs.in-project true

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --without dev

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables according to working configuration
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV PATH="/usr/bin:${PATH}"
ENV PYTHONPATH="/app"
ENV LC_ALL=C

# Verify Tesseract installation and configuration
RUN tesseract --version && \
    tesseract --list-langs && \
    echo "TESSDATA_PREFIX=$TESSDATA_PREFIX" && \
    echo "TESSERACT_CMD=$TESSERACT_CMD" && \
    ls -la $TESSDATA_PREFIX
    ls -la $TESSDATA_PREFIX

# Copy Python environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Start app
CMD .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
