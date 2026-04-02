python:3.12-slim-bookworm
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Environment variables with defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=2

# Start server
CMD uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS