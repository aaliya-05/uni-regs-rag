FROM python:3.11-slim

# tesseract + poppler for the OCR fallback path (see src/ingest.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the indexes at image-build time so container startup is fast.
RUN python -m src.ingest && python -m src.chunk && python -m src.build_index

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
