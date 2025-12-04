FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy and installing all needed requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy of source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Directory for data
RUN mkdir -p ./data

# Expose port for FastAPI
EXPOSE 8001

# Comand for run
CMD ["uvicorn", "src.web.api:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]