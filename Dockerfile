FROM python:3.10-slim

WORKDIR /app

RUN pip install --default-timeout=200 --no-cache-dir --progress-bar off \
    fastapi \
    uvicorn \
    pandas \
    scikit-learn \
    joblib \
    numpy

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
