"""실행 조건(모델·옵션)을 한곳에 모은다. 평가도 이 이름으로 조건을 고른다."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = DATA / "index.sqlite"

EMBED_MODEL = "bge-m3"
TOP_K = 5

# 조건 이름 → Ollama 모델과 옵션
#  - num_gpu: GPU에 올릴 층 수. 0이면 GPU를 전혀 쓰지 않는다(=GPU 없는 사내 PC 흉내).
#  - num_ctx: 한 번에 읽는 글 길이(토큰). 조문 5개 + 질문이면 4096으로 충분하다.
#  - temperature 0: 같은 질문엔 같은 답 — 평가를 다시 돌려도 숫자가 흔들리지 않게.
PROFILES: dict[str, dict] = {
    "gpu": {"model": "qwen3.5:9b", "options": {"num_ctx": 4096, "temperature": 0}},
    "gpu-qwen2.5": {"model": "qwen2.5:7b", "options": {"num_ctx": 4096, "temperature": 0}},
    # 사내 PC 조건: 질문 임베딩까지 CPU에서 돌린다(GPU가 없는 PC이므로).
    "cpu": {"model": "qwen3.5:4b", "options": {"num_ctx": 4096, "temperature": 0, "num_gpu": 0},
            "embed_options": {"num_gpu": 0}},
    # 더 현실적인 사무용 PC 흉내: CPU 스레드를 4개로 묶는다 (이 PC는 8코어 16스레드라 실제보다 빠르게 나온다).
    "cpu-4t": {"model": "qwen3.5:4b",
               "options": {"num_ctx": 4096, "temperature": 0, "num_gpu": 0, "num_thread": 4},
               "embed_options": {"num_gpu": 0, "num_thread": 4}},
}
DEFAULT_PROFILE = "gpu"
