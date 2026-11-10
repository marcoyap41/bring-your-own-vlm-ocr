# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend Runtime
FROM python:3.13-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy python project definition and install dependencies
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r <(uv pip compile pyproject.toml) gunicorn

# Copy backend source
COPY backend/ ./backend/

# Copy built frontend assets from stage 1 into the location backend expects
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

# Use gunicorn with threads for streaming and concurrency
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 8 --timeout 120 backend.main:app
