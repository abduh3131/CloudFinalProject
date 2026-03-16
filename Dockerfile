FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /app/outputs /app/data

# Default: generate sample data and run processing
CMD ["python", "main.py", "--generate-sample", "--output-dir", "outputs"]
