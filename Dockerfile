FROM python:3.10-slim

# Minimal deps — Edge TTS needs no local model or audio tools
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY static ./static
COPY .env .

EXPOSE 5001

# Use gunicorn for production server deployment
CMD gunicorn server:app --bind 0.0.0.0:${PORT:-5001} --workers 2 --timeout 120
