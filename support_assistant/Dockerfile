FROM python:3.12-slim

WORKDIR /app

# System deps needed by chromadb's sqlite/onnxruntime chain on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Populate the ChromaDB collection at build time so `docker run` serves
# requests immediately (also idempotent at container startup via main.py's
# lifespan handler, so this is a build-time speed optimization, not a
# correctness requirement).
RUN python ingest.py

ENV MOCK_LLM=1
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
