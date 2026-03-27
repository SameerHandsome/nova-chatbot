# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — deps-builder
# Installs all Python packages. Cached unless requirements.txt changes.
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS deps-builder

WORKDIR /install

# System deps needed to compile packages (bcrypt, asyncpg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install/packages --no-cache-dir -r requirements.txt


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — frontend-static
# Copies frontend files only. Re-runs only when frontend/ changes.
# ═══════════════════════════════════════════════════════════════════════════════
FROM alpine:3.19 AS frontend-static

WORKDIR /frontend
COPY frontend/ .


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — runtime
# Lean final image: python-slim + deps + app code only.
# No compilers, no build tools, no test files.
# Target size: ~180MB
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r nova && useradd -r -g nova nova

WORKDIR /app

# Runtime system deps only (libpq5 for asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from Stage 1
COPY --from=deps-builder /install/packages /usr/local

# Copy frontend from Stage 2
COPY --from=frontend-static /frontend ./frontend

# Copy backend source (no tests, no evals, no k8s)
COPY backend/ ./backend/

# Set ownership
RUN chown -R nova:nova /app

USER nova

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]