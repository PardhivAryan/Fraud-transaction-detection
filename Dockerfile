# Use official Python image (works on Apple Silicon too)
FROM python:3.11-slim

# Set work directory inside container
WORKDIR /app

# Install system dependencies if needed (optional now)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential && \
#     rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model artifacts
COPY app.py fraud_model_xgb.joblib id_encoders.joblib ./

# Expose port for the API
EXPOSE 8000

# Command to run the app inside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
