FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src DRIFTREV_DATA_DIR=/data
WORKDIR /app
# libgomp1: OpenMP runtime required by LightGBM/XGBoost (missing from the slim base image)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY dashboard ./dashboard
COPY agent ./agent
COPY scripts ./scripts
# non-root user owns the data volume mount point
RUN useradd -m app && mkdir /data && chown app /data
USER app
EXPOSE 8000 8001 8501
CMD ["uvicorn", "driftrev.api:app", "--host", "0.0.0.0", "--port", "8000"]
