FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ls -lah /app/dd_classification_v8/checkpoint-32052

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]