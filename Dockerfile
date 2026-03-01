FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set placeholder environment variables that can be overridden by docker-compose
ENV LOG_LEVEL="INFO"
ENV EMBEDDER_BASE_URL="http://host.docker.internal:11434"
ENV QDRANT_URL="http://qdrant:6333"

# Run the Telegram Bot Gateway by default
CMD ["python", "main.py", "gateway"]
