# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY src/frontend/package*.json ./
RUN npm ci
COPY src/frontend/ .
RUN npm run build

# Stage 2: Python backend + serve built frontend
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than default CUDA build)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ src/

# Copy built frontend into /static so FastAPI can serve it
COPY --from=frontend-builder /app/frontend/dist static/

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
