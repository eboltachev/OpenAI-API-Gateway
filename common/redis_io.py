import json
import os
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urlparse
from uuid import uuid4

from redis.asyncio import Redis

# Значения по умолчанию (могут быть переопределены URL/переменными окружения)
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME") or None
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
REDIS_URL = os.getenv("REDIS_URL") or None  # опционально: redis://[:password]@host:port[/db] или rediss://...

REQUEST_STREAM = os.getenv("REQUEST_STREAM", "openai:requests")
RESPONSE_STREAM_PREFIX = os.getenv("RESPONSE_STREAM_PREFIX", "openai:responses")
RESPONSE_TTL_SEC = int(os.getenv("RESPONSE_TTL_SEC", "300"))

_client: Optional[Redis] = None


def _conn_params() -> Dict[str, Any]:
    """
    Разбирает адрес Redis из REDIS_URL или REDIS_HOST (+ REDIS_PORT, USERNAME, PASSWORD).
    Избавляется от дубля порта вида 'host:port' + port=... и корректно обрабатывает IPv6/схемы.
    """
    username = REDIS_USERNAME
    password = REDIS_PASSWORD
    ssl = False

    # 1) Полный URL имеет приоритет
    if REDIS_URL:
        u = urlparse(REDIS_URL.strip())
        host = u.hostname or REDIS_HOST
        port = u.port or REDIS_PORT
        username = username or (u.username or None)
        password = password or (u.password or None)
        ssl = u.scheme == "rediss"
        return {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "ssl": ssl,
        }

    # 2) Разбор REDIS_HOST: может содержать схему/путь/порт
    h = (REDIS_HOST or "").strip()
    if "://" in h:
        u = urlparse(h)
        host = u.hostname or REDIS_HOST
        port = u.port or REDIS_PORT
        username = username or (u.username or None)
        password = password or (u.password or None)
        ssl = u.scheme == "rediss"
    else:
        host = h
        port = REDIS_PORT
        if host.startswith("["):
            # [IPv6]:port
            end = host.find("]")
            if end != -1:
                ipv6 = host[1:end]
                rest = host[end + 1 :]
                host = ipv6
                if rest.startswith(":"):
                    try:
                        port = int(rest[1:])
                    except Exception:
                        pass
        else:
            if ":" in host:
                # host:port (избавляемся от дубля порта)
                try_host, try_port = host.rsplit(":", 1)
                if try_port.isdigit():
                    host = try_host
                    port = int(try_port)

        # Убрать возможный хвост '/...'
        if "/" in host:
            host = host.split("/", 1)[0]

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "ssl": ssl,
    }


def redis_client() -> Redis:
    global _client
    if _client is None:
        params = _conn_params()
        _client = Redis(
            host=params["host"],
            port=params["port"],
            username=params["username"],
            password=params["password"],
            decode_responses=True,
            ssl=params.get("ssl", False),
        )
    return _client


async def enqueue_request(payload: Dict[str, Any], *, client: Optional[Redis] = None) -> str:
    req_id = payload.get("request_id") or str(uuid4())
    payload = dict(payload, request_id=req_id)
    r = client or redis_client()
    await r.xadd(REQUEST_STREAM, {"json": json.dumps(payload)})
    return req_id


def response_stream_name(request_id: str) -> str:
    return f"{RESPONSE_STREAM_PREFIX}:{request_id}"


async def write_response_raw_json(request_id: str, json_str: str, *, client: Optional[Redis] = None) -> str:
    """Пишем сериализованный JSON (чанк OpenAI/Responses) — для стриминга."""
    r = client or redis_client()
    stream = response_stream_name(request_id)
    msg_id = await r.xadd(stream, {"json": json_str}, maxlen=1000, approximate=True)
    await r.expire(stream, RESPONSE_TTL_SEC)
    return msg_id


async def write_response_object(request_id: str, payload: Dict[str, Any], *, client: Optional[Redis] = None) -> str:
    """Пишем объект — для финального ответа/ошибки/флага done."""
    r = client or redis_client()
    stream = response_stream_name(request_id)
    msg_id = await r.xadd(stream, {"json": json.dumps(payload)}, maxlen=1, approximate=True)
    await r.expire(stream, RESPONSE_TTL_SEC)
    return msg_id


async def wait_for_response(
    request_id: str, timeout_ms: int = 25000, *, client: Optional[Redis] = None
) -> Optional[Dict[str, Any]]:
    r = client or redis_client()
    stream = response_stream_name(request_id)
    events = await r.xread({stream: "0-0"}, count=1, block=timeout_ms)
    if not events:
        return None
    _, msgs = events[0]
    if not msgs:
        return None
    _id, fields = msgs[0]
    data = fields.get("json")
    return json.loads(data) if data else None


async def iter_stream_json(
    request_id: str, *, start_id: str = "0-0", block_ms: int = 15000, client: Optional[Redis] = None
) -> AsyncIterator[Optional[str]]:
    """Итератор json-строки сообщений стрима ответов (для SSE). None -> heartbeat."""
    r = client or redis_client()
    stream = response_stream_name(request_id)
    last_id = start_id
    while True:
        events = await r.xread({stream: last_id}, block=block_ms, count=10)
        if not events:
            yield None
            continue
        _, msgs = events[0]
        for mid, fields in msgs:
            last_id = mid
            yield fields.get("json")
