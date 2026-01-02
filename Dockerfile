# Use Python slim image for smaller container size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry==2.0.1

# Copy only dependency files first for better caching
COPY pyproject.toml poetry.lock* ./

# Configure Poetry: disable virtualenv (use system Python in container)
RUN poetry config virtualenvs.create false

# Install dependencies only (not the project itself)
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Copy application code
COPY src/ ./src/

# Cloud Run uses PORT environment variable
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Use gunicorn as production WSGI server
# Cloud Run sets the PORT env var, we'll use it
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 src.controller:app




