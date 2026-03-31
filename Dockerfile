# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

ARG VERSION=dev
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.title="prismadb"
LABEL org.opencontainers.image.description="Sparse Autoencoder explorer for LLM embeddings"
LABEL org.opencontainers.image.source="https://github.com/edoardoted99/PRISMA_v2"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8000/api/v1/status/ || exit 1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
