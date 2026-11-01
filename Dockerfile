FROM node:24-alpine AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DEVICE=cpu APP_HOST=0.0.0.0 APP_PORT=8092 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements-core.txt ./
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-core.txt
COPY backend/ backend/
COPY data/ data/
COPY results/ results/
COPY --from=web /web/dist frontend/dist/
RUN groupadd -g 10001 support && useradd -u 10001 -g support -M support && mkdir -p .runtime .models .cache && chown -R support:support /app
USER support
EXPOSE 8092
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8092/api/health', timeout=8)"
CMD ["python", "-m", "backend"]
