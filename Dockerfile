FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (build-essential is required for compiling some Python libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application and outputs metadata (excluding large .pkl files which are auto-downloaded on startup)
COPY dashboard ./dashboard
COPY outputs ./outputs

# Expose port (FastAPI dynamic PORT env or defaulting to 7860 on Hugging Face)
EXPOSE 7860

# Run uvicorn server
CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "7860"]
