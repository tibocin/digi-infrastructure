# PCS Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .python-version ./
COPY requirements/ requirements/

# Install Python dependencies
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements/prod.txt

# Copy application code
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

# Create necessary directories
RUN mkdir -p logs storage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Run the application from the src directory
WORKDIR /app/src
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "pcs.main:app", "--host", "0.0.0.0", "--port", "8000"]
