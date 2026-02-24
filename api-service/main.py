import asyncio
import base64
import io
import json
import os
import random
from typing import Any, Awaitable, Callable, Dict

from openai import APIError, AsyncOpenAI, RateLimitError
from redis.asyncio import Redis

from common.async_logging import configure_async_logging
from common.redis_io import REQUEST_STREAM, redis_client, write_response_object, write_response_raw_json

logger = configure_async_logging("api-worker")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_URL = os.getenv("OPENAI_EMBEDDING_URL")
OPENAI_EMBEDDING_KEY = os.getenv("OPENAI_EMBEDDING_KEY")
OPENAI_TRANSCRIPTION_URL = os.getenv("OPENAI_TRANSCRIPTION_URL")
OPENAI_TRANSCRIPTION_KEY = os.getenv("OPENAI_TRANSCRIPTION_KEY")
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "5"))
RETRY_BASE_SEC = float(os.getenv("RETRY_BASE_SEC", "0.5"))
RETRY_CAP_SEC = float(os.getenv("RETRY_CAP_SEC", "8.0"))

if not (OPENAI_BASE_URL and OPENAI_API_KEY):
    logger.warning("OPENAI_CREDS is not set")

client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
client_embeddings = AsyncOpenAI(api_key=OPENAI_EMBEDDING_KEY, base_url=OPENAI_EMBEDDING_URL)
client_transcriptions = AsyncOpenAI(api_key=OPENAI_TRANSCRIPTION_KEY, base_url=OPENAI_TRANSCRIPTION_URL)


def _is_retryable(exc: Exception) -> bool:
    s = str(exc).lower()
    if isinstance(exc, (RateLimitError,)):
        return True
    if isinstance(exc, APIError):
        try:
            code = getattr(exc, "status_code", None)
            if code and int(code) >= 500:
                return True
        except Exception:
            pass
    retry_hints = ("timeout", "temporar", "connection reset", "unavailable", "try again", "eof", "rate limit", "429")
    return any(k in s for k in retry_hints)


async def retry_async(
    fn: Callable[[], Awaitable[Any]],
    *,
    attempts: int = RETRY_ATTEMPTS,
    base: float = RETRY_BASE_SEC,
    cap: float = RETRY_CAP_SEC,
) -> Any:
    last_exc = None
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:
            if not _is_retryable(e):
                raise
            last_exc = e
            sleep_s = min(cap, random.uniform(base, base * (2**i)))
            logger.warning("Retryable error: %s; retrying in %.2fs (attempt %d/%d)", e, sleep_s, i + 1, attempts)
            await asyncio.sleep(sleep_s)
    raise last_exc


def _prepare_openai_args(payload: Dict[str, Any]) -> Dict[str, Any]:
    args = payload.copy()
    for k in ("request_id", "api"):
        args.pop(k, None)
    logger.info(f"{args.get('parameters')=}")
    logger.info(f"{args.get('suffix')=}")
    parameters = args.pop("parameters", {})
    args.pop("suffix", None)
    vendor_keys = ("chat_template", "chat_template_kwargs")
    extra = {k: args.pop(k) for k in vendor_keys if k in args}
    extra.update(parameters)
    if extra:
        eb = dict(args.get("extra_body") or {})
        eb.update(extra)
        args["extra_body"] = eb
    return args


async def _call_once_chat(args: Dict[str, Any]) -> Any:
    a = dict(args)
    a.pop("stream", None)
    return await client.chat.completions.create(**a)


async def _call_once_responses(args: Dict[str, Any]) -> Any:
    a = dict(args)
    a.pop("stream", None)
    return await client.responses.create(**a)


async def _call_once_rerank(args: Dict[str, Any]) -> Any:
    a = dict(args)
    a.pop("stream", None)
    return await client.rerank.create(**a)


async def _stream_chat_completions(args: Dict[str, Any], request_id: str, r: Redis):
    async def _start_stream():
        a = dict(args)
        a["stream"] = True
        return await client.chat.completions.create(**a)
    resp = await retry_async(_start_stream)
    async for chunk in resp:
        try:
            chunk_json = chunk.model_dump_json(exclude_unset=True)
            await write_response_raw_json(request_id, chunk_json, client=r)
        except Exception:
            logger.exception("Chunk serialize error (chat)")
    await write_response_object(request_id, {"done": True}, client=r)


async def _stream_responses_api(args: Dict[str, Any], request_id: str, r: Redis):
    async def _start_stream():
        a = dict(args)
        a["stream"] = True
        return await client.responses.create(**a)
    resp = await retry_async(_start_stream)
    async for event in resp:
        try:
            event_json = event.model_dump_json(exclude_unset=True)
            await write_response_raw_json(request_id, event_json, client=r)
        except Exception:
            logger.exception("Event serialize error (responses)")

    await write_response_object(request_id, {"done": True}, client=r)


async def handle_message(msg_id: str, fields: Dict[str, Any], r: Redis):
    try:
        payload = json.loads(fields.get("json", "{}"))
    except Exception:
        logger.exception("Invalid payload json for message %s", msg_id)
        return
    request_id = payload.get("request_id")
    api = payload.get("api", "chat.completions")
    stream = bool(payload.get("stream", False))
    try:
        args = _prepare_openai_args(payload)
        if stream:
            if api == "responses":
                await _stream_responses_api(args, request_id, r)
            else:
                await _stream_chat_completions(args, request_id, r)
            return
        if api == "models.list":
            comp = await retry_async(lambda: client.models.list())
        elif api == "responses":
            comp = await retry_async(lambda: _call_once_responses(args))
        elif api == "rerank.create":
            comp = await retry_async(lambda: _call_once_rerank(args))
        elif api == "embeddings.create":
            timeout = args.pop("timeout", None)
            async def _call():
                c = client_embeddings.with_options(timeout=timeout) if timeout is not None else client_embeddings
                return await c.embeddings.create(**args)
            comp = await retry_async(_call)
        elif api == "audio.transcriptions.create":
            timeout = args.pop("timeout", None)
            data_b64 = args.pop("file_b64")
            filename = args.pop("filename", "audio.wav")
            uploadable = _uploadable_from_b64(data_b64, filename)
            async def _call():
                c = (
                    client_transcriptions.with_options(timeout=timeout)
                    if timeout is not None
                    else client_transcriptions
                )
                return await c.audio.transcriptions.create(file=uploadable, **args)

            comp = await retry_async(_call)
            result = _normalize_transcription_result(comp)
            await write_response_object(request_id, {"result": result}, client=r)
            return
        else:
            comp = await retry_async(lambda: _call_once_chat(args))
        await write_response_object(request_id, {"result": comp.model_dump()}, client=r)
    except Exception as e:
        logger.exception("Failed to handle message")
        await write_response_object(request_id or "unknown", {"error": str(e)}, client=r)


def _uploadable_from_b64(data_b64: str, filename: str = "audio.wav") -> io.BytesIO:
    bio = io.BytesIO(base64.b64decode(data_b64))
    bio.name = filename
    return bio


def _normalize_transcription_result(obj):
    data = obj.model_dump() if hasattr(obj, "model_dump") else obj
    segs = data.get("segments")
    if isinstance(segs, list):
        norm = []
        for s in segs:
            sd = s if isinstance(s, dict) else getattr(s, "model_dump", lambda: {})()
            seg = {}
            for k in ("start", "end", "text"):
                v = sd.get(k)
                if v is not None:
                    seg[k] = v
            words = sd.get("words") or sd.get("word_segments")
            if isinstance(words, list):
                wlist = []
                for w in words:
                    wd = w if isinstance(w, dict) else getattr(w, "model_dump", lambda: {})()
                    nw = {k: wd.get(k) for k in ("word", "start", "end") if wd.get(k) is not None}
                    if wd.get("score") is not None:
                        nw["score"] = wd["score"]
                    wlist.append(nw)
                seg["words"] = wlist
            norm.append(seg)
        data["segments"] = norm
    return data


async def worker():
    r = redis_client()
    group = os.getenv("REQUEST_GROUP", "api-service")
    consumer = os.getenv("CONSUMER_NAME") or os.uname().nodename
    batch_size = int(os.getenv("REQUEST_BATCH_SIZE", "16"))
    max_parallel = int(os.getenv("MAX_PARALLEL_REQUESTS", "32"))
    semaphore = asyncio.Semaphore(max_parallel)
    try:
        await r.xgroup_create(REQUEST_STREAM, group, id="$", mkstream=True)
        logger.info("Created consumer group '%s' for stream '%s'", group, REQUEST_STREAM)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            logger.exception("xgroup_create failed")
    while True:
        resp = await r.xreadgroup(group, consumer, streams={REQUEST_STREAM: ">"}, count=batch_size, block=30000)
        if not resp:
            continue
        _, messages = resp[0]

        async def _process(mid, fs):
            async with semaphore:
                await handle_message(mid, fs, r)
                await r.xack(REQUEST_STREAM, group, mid)

        await asyncio.gather(*(_process(mid, fs) for mid, fs in messages))

if __name__ == "__main__":
    asyncio.run(worker())
