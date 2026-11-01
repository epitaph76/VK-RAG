FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false HF_HOME=/models
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.lock.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.lock.txt
COPY pyproject.toml ./
COPY siterag ./siterag
COPY benchmarks ./benchmarks
RUN pip install --no-deps . && useradd -m -u 10001 rag && mkdir -p /data /models && chown -R rag:rag /data /models
USER rag
ENV DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "siterag.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--no-proxy-headers"]
