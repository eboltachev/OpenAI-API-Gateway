import json
import os
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import quote, urlparse
from uuid import uuid4

from redis.asyncio import BlockingConnectionPool, Redis


REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME") or None
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
REDIS_URL = os.getenv("REDIS_URL") or None
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

REQUEST_STREAM = os.getenv("REQUEST_STREAM", "openai:requests")
RESPONSE_STREAM_PREFIX = os.getenv("RESPONSE_STREAM_PREFIX", "openai:responses")
RESPONSE_TTL_SEC = int(os.getenv("RESPONSE_TTL_SEC", "300"))
REQUEST_STREAM_MAXLEN = int(os.getenv("REQUEST_STREAM_MAXLEN", "100000"))

REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "512"))
REDIS_POOL_WAIT_TIMEOUT_SEC = int(os.getenv("REDIS_POOL_WAIT_TIMEOUT_SEC", "10"))
REDIS_CONNECT_TIMEOUT_SEC = float(os.getenv("REDIS_CONNECT_TIMEOUT_SEC", "5"))

_regular_client: Optional[Redis] = None
_blocking_client: Optional[Redis] = None


def _normalize_host_and_port() -> tuple[str, int, str]:
    """
    Возвращает host, port, scheme.

    scheme:
    - redis  — обычное TCP-подключение;
    - rediss — TLS-подключение.

    Важно: параметр ssl напрямую в BlockingConnectionPool не передаем,
    потому что в некоторых версиях redis-py это приводит к ошибке:
    AbstractConnection.__init__() got an unexpected keyword argument 'ssl'.
    """
    scheme = "redis"

    if REDIS_URL:
        u = urlparse(REDIS_URL.strip())
        return u.hostname or "", u.port or REDIS_PORT, u.scheme or scheme

    h = (REDIS_HOST or "").strip()

    if "://" in h:
        u = urlparse(h)
        return u.hostname or "", u.port or REDIS_PORT, u.scheme or scheme

    host = h
    port = REDIS_PORT

    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            ipv6 = host[1:end]
            rest = host[end + 1 :]
            host = ipv6

            if rest.startswith(":") and rest[1:].isdigit():
                port = int(rest[1:])
    elif ":" in host:
        maybe_host, maybe_port = host.rsplit(":", 1)
        if maybe_port.isdigit():
            host = maybe_host
            port = int(maybe_port)

    if "/" in host:
        host = host.split("/", 1)[0]

    return host, port, scheme


def _redis_url() -> str:
    """
    Формирует URL подключения к Redis.

    Примеры:
    - redis://user:password@127.0.0.1:6379/0
    - rediss://user:password@redis.example.com:6379/0

    Если REDIS_URL задан явно, он имеет приоритет.
    """
    if REDIS_URL:
        return REDIS_URL.strip()

    host, port, scheme = _normalize_host_and_port()

    if not host:
        raise RuntimeError("REDIS_HOST or REDIS_URL is required")

    auth = ""

    if REDIS_USERNAME and REDIS_PASSWORD:
        auth = f"{quote(REDIS_USERNAME, safe='')}:{quote(REDIS_PASSWORD, safe='')}@"
    elif REDIS_PASSWORD:
        auth = f":{quote(REDIS_PASSWORD, safe='')}@"

    return f"{scheme}://{auth}{host}:{port}/{REDIS_DB}"


def _build_pool(*, socket_timeout: Optional[float]) -> BlockingConnectionPool:
    """
    Создает Redis connection pool через from_url.

    socket_timeout:
    - небольшое число для обычных быстрых команд;
    - None для blocking-команд XREAD/XREADGROUP.
    """
    return BlockingConnectionPool.from_url(
        _redis_url(),
        decode_responses=True,
        max_connections=REDIS_MAX_CONNECTIONS,
        timeout=REDIS_POOL_WAIT_TIMEOUT_SEC,
        socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SEC,
        socket_timeout=socket_timeout,
        health_check_interval=30,
    )


def redis_client() -> Redis:
    """
    Клиент для быстрых Redis-команд:

    - PING
    - XADD
    - XACK
    - EXPIRE
    - XREVRANGE

    Не использовать для долгих blocking-команд XREAD/XREADGROUP.
    """
    global _regular_client

    if _regular_client is None:
        _regular_client = Redis(
            connection_pool=_build_pool(socket_timeout=REDIS_CONNECT_TIMEOUT_SEC)
        )

    return _regular_client


def redis_blocking_client() -> Redis:
    """
    Клиент для blocking-команд Redis Streams:

    - XREAD BLOCK
    - XREADGROUP BLOCK

    socket_timeout=None нужен, чтобы redis-py не обрывал чтение раньше,
    чем истечет Redis BLOCK timeout.
    """
    global _blocking_client

    if _blocking_client is None:
        _blocking_client = Redis(
            connection_pool=_build_pool(socket_timeout=None)
        )

    return _blocking_client


async def close_redis_clients() -> None:
    """
    Корректно закрывает Redis-клиенты.

    Можно вызвать при shutdown FastAPI/worker, если потребуется.
    """
    global _regular_client, _blocking_client

    if _regular_client is not None:
        await _regular_client.aclose()
        _regular_client = None

    if _blocking_client is not None:
        await _blocking_client.aclose()
        _blocking_client = None


async def enqueue_request(payload: Dict[str, Any], *, client: Optional[Redis] = None) -> str:
    """
    Кладет запрос в общий request stream.
    """
    req_id = payload.get("request_id") or str(uuid4())
    payload = dict(payload, request_id=req_id)

    r = client or redis_client()

    await r.xadd(
        REQUEST_STREAM,
        {"json": json.dumps(payload, ensure_ascii=False)},
        maxlen=REQUEST_STREAM_MAXLEN,
        approximate=True,
    )

    return req_id


def response_stream_name(request_id: str) -> str:
    return f"{RESPONSE_STREAM_PREFIX}:{request_id}"


async def write_response_raw_json(
    request_id: str,
    json_str: str,
    *,
    client: Optional[Redis] = None,
) -> str:
    """
    Пишет уже сериализованную JSON-строку в response stream.

    Используется для streaming/SSE чанков.
    """
    r = client or redis_client()
    stream = response_stream_name(request_id)

    msg_id = await r.xadd(stream, {"json": json_str})
    await r.expire(stream, RESPONSE_TTL_SEC)

    return msg_id


async def write_response_object(
    request_id: str,
    payload: Dict[str, Any],
    *,
    client: Optional[Redis] = None,
) -> str:
    """
    Пишет объект ответа в response stream.

    Используется для финальных non-streaming ответов, ошибок и done-флагов.
    """
    r = client or redis_client()
    stream = response_stream_name(request_id)

    msg_id = await r.xadd(
        stream,
        {"json": json.dumps(payload, ensure_ascii=False)},
        maxlen=1,
        approximate=True,
    )

    await r.expire(stream, RESPONSE_TTL_SEC)

    return msg_id


async def wait_for_response(
    request_id: str,
    timeout_ms: int = 25000,
    *,
    client: Optional[Redis] = None,
) -> Optional[Dict[str, Any]]:
    """
    Ожидает первый ответ по request_id.

    Использует отдельный blocking Redis client, чтобы не занимать пул быстрых команд.
    """
    r = client or redis_blocking_client()
    stream = response_stream_name(request_id)

    events = await r.xread(
        {stream: "0-0"},
        count=1,
        block=timeout_ms,
    )

    if not events:
        return None

    _, msgs = events[0]

    if not msgs:
        return None

    _id, fields = msgs[0]
    data = fields.get("json")

    return json.loads(data) if data else None


async def get_response_if_ready(
    request_id: str,
    *,
    client: Optional[Redis] = None,
) -> Optional[Dict[str, Any]]:
    """
    Возвращает последний объект ответа из response stream, если он уже появился.
    """
    r = client or redis_client()
    stream = response_stream_name(request_id)

    events = await r.xrevrange(stream, count=1)

    if not events:
        return None

    _id, fields = events[0]
    data = fields.get("json")

    return json.loads(data) if data else None


async def iter_stream_json(
    request_id: str,
    *,
    start_id: str = "0-0",
    block_ms: int = 15000,
    client: Optional[Redis] = None,
) -> AsyncIterator[Optional[str]]:
    """
    Итератор JSON-строк из response stream.

    Возвращает:
    - str  — очередной JSON-чанк;
    - None — heartbeat, если новых сообщений пока нет.

    Использует отдельный blocking Redis client.
    """
    r = client or redis_blocking_client()
    stream = response_stream_name(request_id)
    last_id = start_id

    while True:
        events = await r.xread(
            {stream: last_id},
            block=block_ms,
            count=10,
        )

        if not events:
            yield None
            continue

        _, msgs = events[0]

        for mid, fields in msgs:
            last_id = mid
            yield fields.get("json")
