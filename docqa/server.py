"""화면(UI)이 쓰는 로컬 HTTP API. 표준 라이브러리만 쓴다(폐쇄망에 추가 설치가 필요 없게).

  GET  /              → ui/index.html (브라우저로 열어도 같은 화면)
  GET  /api/status    → Ollama 연결 여부, 문서 목록, 실행 조건 목록
  POST /api/ask       → {"question": "...", "profile": "cpu"} → 답·근거 조문·걸린 시간

127.0.0.1에만 열어 같은 PC 밖에서는 접근할 수 없다.
SQLite 연결을 여러 스레드가 나눠 쓰지 않도록 요청을 하나씩 처리한다(답변은 어차피 한 번에 하나).
"""
from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from . import config, ollama
from .engine import Engine

UI_DIR = config.ROOT / "ui"
_engines: dict[str, Engine] = {}


def engine(profile: str) -> Engine:
    if profile not in _engines:
        _engines[profile] = Engine(profile)
    return _engines[profile]


def status() -> dict:
    try:
        loaded = [m["name"] for m in ollama.loaded()]
        ok = True
    except Exception:
        loaded, ok = [], False
    eng = engine(config.DEFAULT_PROFILE)
    return {
        "ollama": ok,
        "loaded_models": loaded,
        "docs": [{"doc_id": k, "title": v} for k, v in sorted(eng.titles.items())],
        "profiles": [{"name": k, "model": v["model"], "gpu": v["options"].get("num_gpu") != 0}
                     for k, v in config.PROFILES.items()],
        "default_profile": config.DEFAULT_PROFILE,
    }


def ask(question: str, profile: str) -> dict:
    eng = engine(profile)
    a = eng.ask(question)
    by_id = {c["chunk_id"]: c for c in a.contexts}
    return {
        "question": question,
        "profile": profile,
        "model": eng.profile["model"],
        "answerable": a.answerable,
        "answer": a.answer,
        "citations": [{"chunk_id": cid, "doc_title": eng.titles.get(by_id[cid]["doc_id"], ""),
                       "article": by_id[cid]["article"], "title": by_id[cid]["title"],
                       "text": by_id[cid]["text"]} for cid in a.citations],
        "retrieved": [{"chunk_id": c["chunk_id"], "doc_title": eng.titles.get(c["doc_id"], ""),
                       "article": c["article"], "title": c["title"]} for c in a.contexts],
        "timing": {k: round(v, 2) if isinstance(v, float) else v for k, v in a.timing.items()},
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")   # Tauri 창(다른 출처)에서 부르기 위해
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        if self.path == "/api/status":
            return self._json(200, status())
        rel = "index.html" if self.path in ("/", "") else self.path.lstrip("/").split("?")[0]
        f = (UI_DIR / rel).resolve()
        if UI_DIR.resolve() not in f.parents or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        if self.path != "/api/ask":
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            q = (body.get("question") or "").strip()
            profile = body.get("profile") or config.DEFAULT_PROFILE
            if not q:
                return self._json(400, {"error": "질문이 비어 있습니다."})
            if profile not in config.PROFILES:
                return self._json(400, {"error": f"알 수 없는 실행 조건: {profile}"})
            self._json(200, ask(q, profile))
        except Exception as e:   # 화면에 원인을 보여 준다 (예: Ollama가 꺼져 있음)
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        print("[serve]", fmt % args, flush=True)


def _exit_with_parent(pid: int) -> None:
    """앱(부모)이 꺼지면 엔진도 스스로 끈다. 앱이 강제 종료되어도 엔진이 남지 않게."""
    import os
    import threading
    import time

    import psutil

    def watch():
        while psutil.pid_exists(pid):
            time.sleep(1)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def serve(port: int = 8765, parent_pid: int | None = None) -> None:
    if parent_pid:
        _exit_with_parent(parent_pid)
    httpd = HTTPServer(("127.0.0.1", port), Handler)
    print(f"docqa API: http://127.0.0.1:{port}  (Ctrl+C로 종료)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
