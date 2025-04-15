FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up model directories
RUN mkdir -p /app/models/ct_report

# Copy application code
COPY api/ ./api/
COPY models/ ./models/
COPY services/ ./services/
COPY utils/ ./utils/
COPY config.py .
COPY main.py .

# Create upload directory
RUN mkdir -p uploads

# Expose the port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV USE_8BIT_QUANTIZATION=True
ENV ENABLE_CORS=True
ENV ALLOW_SIMULATED_MODELS=True

# Command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]