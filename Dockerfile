FROM python:3.11-slim

WORKDIR /fotbot_app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (except ignored)
COPY . .

# Run application
CMD ["python", "main.py"]
