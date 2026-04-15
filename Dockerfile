FROM python:3.11-slim

WORKDIR /app

# curl is needed for the API health-check in docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Pre-create runtime directories so volume mounts land cleanly
RUN mkdir -p outputs logs rag/chroma_db eval/reports

# Ports used by the two services (documentation only — published in compose)
EXPOSE 8000 8501
