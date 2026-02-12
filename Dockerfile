FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Optional build-time arg to download models/artifacts archive (tar.gz or zip)
ARG MODEL_ARCHIVE_URL=""
ENV MODEL_ARCHIVE_URL=${MODEL_ARCHIVE_URL}

# Install system deps required by some packages and for fetching archives
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps early for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application sources
COPY . /app

# If a MODEL_ARCHIVE_URL is provided, download and extract into the project
RUN if [ -n "${MODEL_ARCHIVE_URL}" ]; then \
      echo "Downloading model archive from ${MODEL_ARCHIVE_URL}" && \
      curl -fsSL "${MODEL_ARCHIVE_URL}" -o /tmp/models_archive && \
      mkdir -p /app/models /app/artifacts && \
      if file /tmp/models_archive | grep -q zip; then \
          unzip -q /tmp/models_archive -d /app/; \
      else \
          tar -xzf /tmp/models_archive -C /app/ || true; \
      fi && rm -f /tmp/models_archive; \
    else \
      echo "No MODEL_ARCHIVE_URL provided; relying on repo files."; \
    fi

# Expose Streamlit port
EXPOSE 8501

# Default command runs the main Streamlit app
CMD ["streamlit", "run", "detection_car_price/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
