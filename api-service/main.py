import asyncio
import base64
import io
import json
import os
import random
from typing import Any, Awaitable, Callable, Dict, List, Tuple

import httpx
from openai import APIError, AsyncOpenAI, RateLimitError
from redis.asyncio import Redis

from common.async_logging import configure_async_logging
from common.redis_io import REQUEST_STREAM, redis_client, write_response_object, write_response_raw_json

logger = configure_async_logging("api-worker")

RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "5"))
RETRY_BASE_SEC = float(os.getenv("RETRY_BASE_SEC", "0.5"))
RETRY_CAP_SEC = float(os.getenv("RETRY_CAP_SEC", "8.0"))

_CLIENT_CACHE: Dict[Tuple[str, str], AsyncOpenAI] = {}
_HTTP_CLIENT_CACHE: Dict[Tuple[str, str], httpx.AsyncClient] = {}


def _get_client(base_url: str | None, api_key: str | None) -> AsyncOpenAI:
    key = ((base_url or "").strip(), (api_key or "").strip())
    if key not in _CLIENT_CACHE:
        _HTTP_CLIENT_CACHE[key] = httpx.AsyncClient(verify=False)
        _CLIENT_CACHE[key] = AsyncOpenAI(api_key=key[1], base_url=key[0], http_client=_HTTP_CLIENT_CACHE[key])
    return _CLIENT_CACHE[key]


def _load_model_routing(env_var: str) -> Dict[str, AsyncOpenAI]:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return {}
    try:
        cfg = json.loads(raw)
    except Exception:
        logger.exception("Invalid JSON in %s", env_var)
        return {}
    if not isinstance(cfg, dict):
        logger.warning("%s must be a JSON object", env_var)
        return {}
    routes: Dict[str, AsyncOpenAI] = {}
    for model, conf in cfg.items():
        if not isinstance(model, str) or not isinstance(conf, dict):
            continue
        if not conf.get("base_url"):
            logger.warning("Missing base_url for %s[%s]", env_var, model)
            continue
        routes[model] = _get_client(conf.get("base_url"), conf.get("api_key"))
    return routes


chat_routes = _load_model_routing("OPENAI_CHAT_MODEL_ROUTES")
embedding_routes = _load_model_routing("OPENAI_EMBEDDING_MODEL_ROUTES")
reranking_routes = _load_model_routing("OPENAI_RERANKING_MODEL_ROUTES")
transcription_routes = _load_model_routing("OPENAI_TRANSCRIPTION_MODEL_ROUTES")


def _client_for_model(args: Dict[str, Any], routes: Dict[str, AsyncOpenAI], route_name: str) -> AsyncOpenAI:
    model = args.get("model")
    if isinstance(model, str):
        if model in routes:
            return routes[model]
        if "*" in routes:
            return routes["*"]
        raise ValueError(f"Model '{model}' is not configured in {route_name}")
    if "*" in routes:
        return routes["*"]
    raise ValueError(f"Field 'model' is required for {route_name}")


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
    c = _client_for_model(a, chat_routes, "OPENAI_CHAT_MODEL_ROUTES")
    return await c.chat.completions.create(**a)


async def _call_once_responses(args: Dict[str, Any]) -> Any:
    a = dict(args)
    a.pop("stream", None)
    c = _client_for_model(a, chat_routes, "OPENAI_CHAT_MODEL_ROUTES")
    return await c.responses.create(**a)


async def _call_once_rerank(args: Dict[str, Any]) -> Any:
    a = dict(args)
    a.pop("stream", None)
    c = _client_for_model(a, reranking_routes, "OPENAI_RERANKING_MODEL_ROUTES")
    # Use relative path so provider-specific base paths like `/v1` are preserved.
    return await c.post("rerank", cast_to=Dict[str, Any], body=a)


def _extract_models_data(payload: Any) -> List[Any]:
    data = payload.get("data", []) if isinstance(payload, dict) else []
    return list(data) if isinstance(data, list) else []


def _normalize_model_obj(item: Any) -> Dict[str, Any] | None:
    if isinstance(item, dict):
        mid = item.get("id")
        return item if isinstance(mid, str) else None
    if isinstance(item, str):
        return {"id": item, "object": "model"}
    return None


async def _list_models_combined() -> Dict[str, Any]:
    clients = {*chat_routes.values(), *embedding_routes.values(), *reranking_routes.values(), *transcription_routes.values()}
    merged: Dict[str, Dict[str, Any]] = {}
    last_error = None
    for c in clients:
        try:
            # Use raw HTTP call to avoid strict SDK typing issues when providers
            # return `data` as strings instead of OpenAI Model objects.
            part = await retry_async(lambda: c.get("/models", cast_to=Dict[str, Any]))
            for item in _extract_models_data(part):
                model_obj = _normalize_model_obj(item)
                if model_obj is None:
                    continue
                merged[model_obj["id"]] = model_obj
        except Exception as e:
            last_error = e
            logger.warning("models.list failed for one route: %s", e)
    if merged:
        return {"object": "list", "data": list(merged.values())}
    if last_error:
        raise last_error
    return {"object": "list", "data": []}


def _dump_result(obj: Any) -> Any:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


async def _stream_chat_completions(args: Dict[str, Any], request_id: str, r: Redis):
    async def _start_stream():
        a = dict(args)
        a["stream"] = True
        c = _client_for_model(a, chat_routes, "OPENAI_CHAT_MODEL_ROUTES")
        return await c.chat.completions.create(**a)
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
        c = _client_for_model(a, chat_routes, "OPENAI_CHAT_MODEL_ROUTES")
        return await c.responses.create(**a)
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
            comp = await _list_models_combined()
        elif api == "responses":
            comp = await retry_async(lambda: _call_once_responses(args))
        elif api == "rerank.create":
            logger.info(f"{args=}")
            comp = await retry_async(lambda: _call_once_rerank(args))
        elif api == "embeddings.create":
            timeout = args.pop("timeout", None)

            async def _call():
                base_client = _client_for_model(args, embedding_routes, "OPENAI_EMBEDDING_MODEL_ROUTES")
                c = base_client.with_options(timeout=timeout) if timeout is not None else base_client
                # Use low-level POST to preserve provider-specific fields that are not
                # part of the OpenAI SDK embeddings.create() typed signature.
                return await c.post("/embeddings", cast_to=Dict[str, Any], body=args)
            comp = await retry_async(_call)
        elif api == "audio.transcriptions.create":
            timeout = args.pop("timeout", None)
            data_b64 = args.pop("file_b64")
            filename = args.pop("filename", "audio.wav")
            uploadable = _uploadable_from_b64(data_b64, filename)
            async def _call():
                base_client = _client_for_model(args, transcription_routes, "OPENAI_TRANSCRIPTION_MODEL_ROUTES")
                c = (
                    base_client.with_options(timeout=timeout)
                    if timeout is not None
                    else base_client
                )
                return await c.audio.transcriptions.create(file=uploadable, **args)

            comp = await retry_async(_call)
            result = _normalize_transcription_result(comp)
            await write_response_object(request_id, {"result": result}, client=r)
            return
        else:
            comp = await retry_async(lambda: _call_once_chat(args))
        await write_response_object(request_id, {"result": _dump_result(comp)}, client=r)
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
    in_flight: set[asyncio.Task] = set()

    async def _process(mid: str, fs: Dict[str, Any]):
        async with semaphore:
            try:
                await handle_message(mid, fs, r)
            finally:
                await r.xack(REQUEST_STREAM, group, mid)

    def _track(task: asyncio.Task):
        in_flight.add(task)

    async def _wait_for_slot_if_needed():
        if len(in_flight) < max_parallel:
            return
        done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                task.result()
            finally:
                in_flight.discard(task)

    async def _drain_finished():
        finished = [task for task in in_flight if task.done()]
        for task in finished:
            try:
                task.result()
            finally:
                in_flight.discard(task)

    try:
        await r.xgroup_create(REQUEST_STREAM, group, id="$", mkstream=True)
        logger.info("Created consumer group '%s' for stream '%s'", group, REQUEST_STREAM)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            logger.exception("xgroup_create failed")
    while True:
        await _wait_for_slot_if_needed()
        await _drain_finished()
        resp = await r.xreadgroup(group, consumer, streams={REQUEST_STREAM: ">"}, count=batch_size, block=30000)
        if not resp:
            continue
        _, messages = resp[0]
        for mid, fs in messages:
            await _wait_for_slot_if_needed()
            task = asyncio.create_task(_process(mid, fs))
            _track(task)

if __name__ == "__main__":
    asyncio.run(worker())
