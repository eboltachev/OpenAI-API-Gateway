FROM python:3.12-slim
WORKDIR /app
COPY endpoint-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common ./common
COPY endpoint-service/main.py .
EXPOSE ${API_ENDPOINT_PORT}
CMD uvicorn main:app --host ${API_ENDPOINT_HOST} --port ${API_ENDPOINT_PORT}
