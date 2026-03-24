import asyncio
import base64
import json
import os
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, model_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, StreamingResponse

from common.async_logging import configure_async_logging
from common.redis_io import enqueue_request, get_response_if_ready, iter_stream_json, redis_client, wait_for_response

app = FastAPI(title="API")
logger = configure_async_logging("endpoint-service")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for path %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


class RequestBody(BaseModel):
    model_config = {"extra": "allow"}


class EmbeddingsBody(RequestBody):
    model: str
    input: Optional[Any] = None
    messages: Optional[Any] = None
    timeout_ms: Optional[int] = None

    @model_validator(mode="after")
    def ensure_input_exists(self):
        if self.input is None and self.messages is not None:
            self.input = self.messages
        if self.input is None:
            raise ValueError("Field 'input' is required (or provide 'messages' as compatibility alias).")
        return self


class RerankBody(RequestBody):
    model: str
    query: str
    documents: list[Any]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = None
    max_chunks_per_doc: Optional[int] = None


def _unauthorized(msg: str, code: str):
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": msg,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
    )


async def _auth_middleware(request: Request, call_next):
    ENDPOINT_API_KEY = os.getenv("ENDPOINT_API_KEY", "")
    p = request.url.path
    if p in ("/health", "/healthz") or p.startswith("/docs") or p.startswith("/openapi"):
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)
    if ENDPOINT_API_KEY:
        auth = request.headers.get("authorization") or ""
        token = None
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
        token = token or request.headers.get("x-api-key") or request.headers.get("api-key")
        if not token:
            return _unauthorized("You must provide an API key.", "missing_api_key")
        if token != ENDPOINT_API_KEY:
            logger.warning("Invalid API key from %s", request.client.host if request.client else "unknown")
            return _unauthorized("Incorrect API key provided.", "invalid_api_key")
    return await call_next(request)


app.add_middleware(BaseHTTPMiddleware, dispatch=_auth_middleware)


@app.get("/health")
async def health():
    r = redis_client()
    try:
        pong = await r.ping()
        return {"status": "ok", "redis": bool(pong)}
    except Exception:
        logger.exception("Health check failed")
        return {"status": "degraded", "redis": False}


# ---- Models API ----
@app.get("/v1/models")
async def list_models():
    request_id = str(uuid4())
    payload = {"request_id": request_id, "api": "models.list"}
    await enqueue_request(payload)
    timeout_ms = int(os.getenv("SYNC_TIMEOUT_MS", "25000"))
    resp = await wait_for_response(request_id, timeout_ms=timeout_ms)
    if not resp:
        raise HTTPException(status_code=504, detail="Timeout waiting for models list")
    if "error" in resp:
        logger.error("Models backend error: %s", resp["error"])
        raise HTTPException(status_code=502, detail=resp["error"])
    return resp.get("result", resp)


@app.post("/v1/embeddings")
async def get_embeddings(body: EmbeddingsBody):
    body_data = body.model_dump(exclude_none=True, exclude={"messages"})
    request_id = str(uuid4())
    payload = dict(body_data, request_id=request_id, api="embeddings.create")
    async_mode = _is_async_requested(payload)
    timeout_ms = body_data.get("timeout_ms")
    sync_timeout_ms = timeout_ms or int(os.getenv("SYNC_TIMEOUT_MS", "25000"))
    if timeout_ms is not None:
        payload["timeout"] = timeout_ms / 1000.0  # сек для SDK
    await enqueue_request(payload)
    if async_mode:
        return _async_accepted_response(request_id)
    resp = await wait_for_response(request_id, timeout_ms=sync_timeout_ms)
    if not resp:
        raise HTTPException(status_code=504, detail="Timeout waiting for embeddings")
    if "error" in resp:
        logger.error("Embeddings backend error: %s", resp["error"])
        raise HTTPException(status_code=502, detail=resp["error"])
    return resp.get("result", resp)




def _is_async_requested(payload: Dict[str, Any]) -> bool:
    return bool(payload.pop("async", False) or payload.pop("background", False))


def _async_accepted_response(request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=202,
        content={
            "id": request_id,
            "object": "async.request",
            "status": "queued",
            "status_endpoint": f"/v1/requests/{request_id}",
        },
    )

def _adapt_chat_chunk_to_completions(obj: Dict[str, Any], fallback_model: Any) -> Optional[Dict[str, Any]]:
    choices = obj.get("choices", [])
    out_choices = []
    has_payload = False

    for i, ch in enumerate(choices):
        txt = ""
        if isinstance(ch, dict):
            delta = ch.get("delta") or {}
            if isinstance(delta, dict):
                txt = delta.get("content") or ""
            # some providers may stream full message/content instead of delta
            if not txt:
                msg = ch.get("message") or {}
                txt = msg.get("content") or ""

        finish_reason = ch.get("finish_reason")
        if txt or finish_reason:
            has_payload = True

        out_choices.append(
            {
                "text": txt or "",
                "index": ch.get("index", i),
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        )

    if not has_payload:
        return None

    return {
        "id": obj.get("id"),
        "object": "text_completion.chunk",
        "created": obj.get("created"),
        "model": obj.get("model") or fallback_model,
        "choices": out_choices,
    }


async def _proxy(body: Dict[str, Any], api_name: str, stream_format: Optional[str] = None, fallback_model: Any = None):
    request_id = str(uuid4())
    payload = dict(body)
    payload["request_id"] = request_id
    payload["api"] = api_name
    stream = bool(payload.get("stream", False))
    async_mode = _is_async_requested(payload)

    if async_mode and stream:
        raise HTTPException(status_code=400, detail="Async mode is not supported with stream=true")

    await enqueue_request(payload)

    if async_mode:
        return _async_accepted_response(request_id)

    if not stream:
        timeout_ms = int(os.getenv("SYNC_TIMEOUT_MS", "50000"))
        resp = await wait_for_response(request_id, timeout_ms=timeout_ms)
        if not resp:
            raise HTTPException(status_code=504, detail="Timeout waiting for completion")
        if "error" in resp:
            msg = str(resp["error"])
            lowered = msg.lower()
            if "rate" in lowered or "429" in lowered:
                raise HTTPException(status_code=429, detail=msg)
            if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
                raise HTTPException(status_code=401, detail=msg)
            raise HTTPException(status_code=502, detail=msg)
        return resp.get("result", resp)

    async def sse():
        async for j in iter_stream_json(request_id):
            try:
                if j is None:
                    yield ": keepalive\n\n"
                    continue
                obj = json.loads(j)

                if obj.get("done") is True:
                    yield "data: [DONE]\n\n"
                    return

                if obj.get("error"):
                    yield f"data: {json.dumps({'error': obj['error']}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if stream_format == "completions":
                    mapped = _adapt_chat_chunk_to_completions(obj, fallback_model=fallback_model)
                    if mapped is None:
                        # skip chunks that don't contain text or finish_reason
                        continue
                    yield f"data: {json.dumps(mapped, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {j}\n\n"

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception("SSE error in _proxy")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)


@app.post("/v1/chat/completions")
async def chat_completions(body: RequestBody):
    return await _proxy(body.model_dump(exclude_none=True), "chat.completions")


def _to_chat_from_completions(body: Dict[str, Any]) -> Dict[str, Any]:
    b = dict(body)
    prompt = b.pop("prompt", None)
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    if prompt is not None and "messages" not in b:
        b["messages"] = [{"role": "user", "content": str(prompt)}]
    return b


def _chat_to_completions_response(resp: Dict[str, Any], fallback_model: Any) -> Dict[str, Any]:
    choices = resp.get("choices", [])
    out_choices = []
    for i, ch in enumerate(choices):
        content = ""
        if isinstance(ch, dict):
            msg = ch.get("message") or {}
            content = msg.get("content") or ""
        out_choices.append(
            {
                "text": content,
                "index": ch.get("index", i) if isinstance(ch, dict) else i,
                "logprobs": None,
                "finish_reason": (ch.get("finish_reason") if isinstance(ch, dict) else None),
            }
        )
    return {
        "id": resp.get("id"),
        "object": "text_completion",
        "created": resp.get("created"),
        "model": resp.get("model") or fallback_model,
        "choices": out_choices,
        "usage": resp.get("usage"),
    }


@app.post("/v1/completions")
async def completions(body: RequestBody):
    body_data = body.model_dump(exclude_none=True)
    chat_body = _to_chat_from_completions(body_data)

    # Если stream=True, проксируем с адаптацией потокового формата в completions
    if bool(chat_body.get("stream", False)):
        resp = await _proxy(
            chat_body,
            "chat.completions",
            stream_format="completions",
            fallback_model=body_data.get("model"),
        )
        return resp

    # Синхронный ответ: проксируем и конвертируем результат
    resp = await _proxy(chat_body, "chat.completions")
    if isinstance(resp, (StreamingResponse, JSONResponse)):
        return resp  # для stream и async-accepted веток

    result = _chat_to_completions_response(resp, fallback_model=body_data.get("model"))
    return JSONResponse(result)


# ---- Responses API ----
@app.post("/v1/responses")
async def responses_api(body: RequestBody):
    request_id = str(uuid4())
    payload = body.model_dump(exclude_none=True)
    payload["request_id"] = request_id
    payload["api"] = "responses"
    stream = bool(payload.get("stream", False))
    async_mode = _is_async_requested(payload)
    if async_mode and stream:
        raise HTTPException(status_code=400, detail="Async mode is not supported with stream=true")
    await enqueue_request(payload)
    if async_mode:
        return _async_accepted_response(request_id)
    if not stream:
        timeout_ms = int(os.getenv("SYNC_TIMEOUT_MS", "25000"))
        resp = await wait_for_response(request_id, timeout_ms=timeout_ms)
        if not resp:
            raise HTTPException(status_code=504, detail="Timeout waiting for completion")
        if "error" in resp:
            msg = str(resp["error"])
            lowered = msg.lower()
            if "rate" in lowered or "429" in lowered:
                raise HTTPException(status_code=429, detail=msg)
            if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
                raise HTTPException(status_code=401, detail=msg)
            raise HTTPException(status_code=502, detail=msg)
        return resp.get("result", resp)

    async def sse():
        async for j in iter_stream_json(request_id):
            try:
                if j is None:
                    yield ": keepalive\n\n"
                    continue
                obj = json.loads(j)
                if obj.get("done") is True:
                    yield "data: [DONE]\n\n"
                    return
                if obj.get("error"):
                    yield f"event: error\ndata: {json.dumps({'error': obj['error']}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                ev_type = obj.get("type") or "message"
                yield f"event: {ev_type}\n"
                yield f"data: {j}\n\n"
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception("SSE error in responses_api")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
                return

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)


@app.post("/v1/audio/transcriptions")
async def get_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("openai/whisper-large-v3"),
    language: Optional[str] = Form("ru"),
    prompt: Optional[str] = Form(""),
    temperature: float | None = Form(0.0),
    response_format: Optional[str] = Form("verbose_json"),
    timestamp_granularities: Optional[str] = Form("word"),
    timeout_ms: int | None = Form(None),
    async_request: bool = Form(False),
):
    request_id = str(uuid4())
    raw = await file.read()
    file_b64 = base64.b64encode(raw).decode("ascii")
    payload: Dict[str, Any] = {
        "request_id": request_id,
        "api": "audio.transcriptions.create",
        "model": model,
        "file_b64": file_b64,
        "filename": file.filename or "audio.wav",
    }
    if language is not None:
        payload["language"] = language
    if prompt is not None:
        payload["prompt"] = prompt
    if response_format is not None:
        payload["response_format"] = response_format
    if temperature is not None:
        payload["temperature"] = temperature
    if timestamp_granularities:
        payload["timestamp_granularities"] = [s.strip() for s in timestamp_granularities.split(",") if s.strip()]
    if timeout_ms is not None:
        try:
            timeout_ms = int(timeout_ms)
        except Exception:
            logger.exception("Invalid timeout_ms for transcriptions")
            timeout_ms = None
    sync_timeout_ms = timeout_ms or int(os.getenv("SYNC_TIMEOUT_MS", "25000"))
    if timeout_ms is not None:
        payload["timeout"] = timeout_ms / 1000.0
    await enqueue_request(payload)
    if async_request:
        return _async_accepted_response(request_id)
    resp = await wait_for_response(request_id, timeout_ms=sync_timeout_ms)
    if not resp:
        raise HTTPException(status_code=504, detail="Timeout waiting for transcription")
    if "error" in resp:
        logger.error("Transcriptions backend error: %s", resp["error"])
        raise HTTPException(status_code=502, detail=resp["error"])
    return resp.get("result", resp)


@app.get("/v1/requests/{request_id}")
async def get_async_request_result(request_id: str):
    resp = await get_response_if_ready(request_id)
    if not resp:
        return JSONResponse(status_code=202, content={"id": request_id, "status": "queued"})
    if "error" in resp:
        return JSONResponse(status_code=200, content={"id": request_id, "status": "failed", "error": resp["error"]})
    return JSONResponse(status_code=200, content={"id": request_id, "status": "completed", "result": resp.get("result", resp)})


@app.post("/v1/rerank")
async def rerank(body: RerankBody):
    body_data = body.model_dump(exclude_none=True)
    return await _proxy(body_data, "rerank.create")
