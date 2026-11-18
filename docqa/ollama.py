"""Ollama 로컬 서버(http://localhost:11434) 호출. 인터넷은 쓰지 않는다."""
from __future__ import annotations

import os

import httpx

HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
_client = httpx.Client(base_url=HOST, timeout=httpx.Timeout(600.0))


def embed(model: str, texts: list[str], options: dict | None = None) -> list[list[float]]:
    r = _client.post("/api/embed", json={"model": model, "input": texts, "keep_alive": "30m",
                                         "options": options or {}})
    r.raise_for_status()
    return r.json()["embeddings"]


def chat(model: str, messages: list[dict], *, schema: dict | None = None,
         options: dict | None = None, think: bool | None = False) -> dict:
    body: dict = {"model": model, "messages": messages, "stream": False,
                  "options": options or {}, "keep_alive": "30m"}
    if schema is not None:
        body["format"] = schema
    if think is not None:
        body["think"] = think
    r = _client.post("/api/chat", json=body)
    if r.status_code == 400 and "think" in r.text:   # 생각 모드가 없는 모델
        body.pop("think", None)
        r = _client.post("/api/chat", json=body)
    r.raise_for_status()
    return r.json()


def unload(model: str) -> None:
    """메모리에서 모델을 내린다 (조건을 바꿔 잴 때 앞 조건의 흔적을 지우려고)."""
    _client.post("/api/generate", json={"model": model, "keep_alive": 0})


def loaded() -> list[dict]:
    return _client.get("/api/ps").json().get("models", [])
