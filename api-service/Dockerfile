FROM python:3.12-slim
WORKDIR /app
COPY api-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common ./common
COPY api-service/main.py .
CMD ["python", "main.py"]
