FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000 8001

# Default command launches the unified Stage 4 Streamlit Dashboard
CMD ["streamlit", "run", "src/stage4_orchestration/app_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]

