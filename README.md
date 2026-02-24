# OpenAI API Proxy

Прокси из двух сервисов:
- `endpoint-service` — OpenAI-совместимый HTTP endpoint для клиентов.
- `api-service` — worker, который читает задачи из Redis и вызывает upstream OpenAI-совместимые API (chat/models/responses/rerank/embeddings/transcriptions), в том числе на разных хостах/портах.

## Архитектура
1. Клиент вызывает `endpoint-service` по OpenAI API.
2. Endpoint пишет запрос в Redis stream `REQUEST_STREAM`.
3. `api-service` читает stream consumer-группой, вызывает upstream и пишет результат в per-request stream ответа.
4. Endpoint ждёт sync-ответ или стримит SSE-чанки клиенту.

## Быстрый старт
```bash
docker compose -f compose-api.yml up --build -d
docker compose -f compose-endpoint.yml up --build -d
```

## Важные переменные окружения
### Общие / Redis
- `REDIS_URL` (предпочтительно) или `REDIS_HOST` + `REDIS_PORT`
- `REDIS_USERNAME`, `REDIS_PASSWORD`
- `REQUEST_STREAM` (default: `openai:requests`)
- `RESPONSE_STREAM_PREFIX` (default: `openai:responses`)
- `RESPONSE_TTL_SEC` (default: `300`)
- `SYNC_TIMEOUT_MS` (default: `25000`/`50000` в зависимости от endpoint)

### endpoint-service
- `ENDPOINT_API_KEY` — ключ для входящих клиентских запросов.
- `LOG_LEVEL` — уровень логирования.

### api-service
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_URL`, `OPENAI_EMBEDDING_KEY`
- `OPENAI_TRANSCRIPTION_URL`, `OPENAI_TRANSCRIPTION_KEY`
- `REQUEST_GROUP` (default: `api-service`)
- `CONSUMER_NAME` (optional)
- `REQUEST_BATCH_SIZE` (default: `16`)
- `MAX_PARALLEL_REQUESTS` (default: `32`)
- `RETRY_ATTEMPTS`, `RETRY_BASE_SEC`, `RETRY_CAP_SEC`
- `LOG_LEVEL`

## Примечания
- Логирование в обоих сервисах асинхронное (QueueHandler/QueueListener).
- Для потоковых ответов stream не обрезается во время генерации; очистка происходит по `RESPONSE_TTL_SEC`, чтобы не терять SSE-чанки у медленных клиентов.
