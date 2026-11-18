"""평가 세트를 한 조건(profile)으로 돌려 숫자를 낸다.

지표 (모두 기계 채점 — 사람 판단이 끼지 않아 다시 돌려도 같은 숫자):
- 정확도: 답이 있는 질문 중, 모델이 답했고 정답 핵심어가 모두 답에 들어 있는 비율.
- 근거 일치율: 답이 있는 질문 중, 모델이 인용한 조문에 정답 근거 조문이 들어 있는 비율.
- «모른다» 정답률: 근거 없는 질문 중, 모델이 «모른다»고 답한 비율.
- 잘못된 거절: 답이 있는 질문인데 «모른다»고 한 비율 (모른다를 남발하는지 확인).
- 검색 재현율@5: 정답 조문이 검색 상위 5개에 들어온 비율 (LLM 탓인지 검색 탓인지 나누려고).
- 응답 시간: 질문 하나당 검색+생성 시간 (평균·중앙값·95% 지점).
- 최대 메모리: Ollama 프로세스들 + 이 파이썬 프로세스의 RAM 합(RSS) 최댓값, GPU 메모리 최댓값.
  0.2초마다 잰다. 첫 질문 전에 모델을 내려 두어, 모델을 올리는 순간부터 포함한다.

사용: uv run python -m evaluation.run_eval --profile cpu [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from docqa import config, ollama
from docqa.engine import Engine

HERE = Path(__file__).resolve().parent


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def base_id(cid: str) -> str:
    return cid.split("#", 1)[0]   # 긴 조를 나눈 조각(#2)도 같은 조로 본다


class MemoryMonitor(threading.Thread):
    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        self.interval, self.stop_flag = interval, threading.Event()
        self.peak_ram = 0
        self.peak_vram = 0
        self.by_proc: dict[str, int] = {}   # «이름:pid» → 최대 RSS (무엇이 메모리를 쓰는지 보려고)
        self.me = psutil.Process(os.getpid())

    @staticmethod
    def vram_used_mib() -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            return int(out.strip().splitlines()[0])
        except Exception:
            return 0

    def ram_bytes(self) -> int:
        total = self.me.memory_info().rss
        for p in psutil.process_iter(["name"]):
            try:
                # Ollama 0.40부터 모델은 llama-server.exe라는 별도 프로세스에서 돈다.
                if (p.info["name"] or "").lower().startswith(("ollama", "llama-server")):
                    r = p.memory_info().rss
                    total += r
                    k = f"{p.info['name']}:{p.pid}"
                    self.by_proc[k] = max(self.by_proc.get(k, 0), r)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def run(self):
        n = 0
        while not self.stop_flag.is_set():
            self.peak_ram = max(self.peak_ram, self.ram_bytes())
            if n % 5 == 0:                      # nvidia-smi는 느려서 1초마다
                self.peak_vram = max(self.peak_vram, self.vram_used_mib())
            n += 1
            time.sleep(self.interval)


def load_set() -> list[dict]:
    items = [json.loads(l) for l in open(HERE / "qa_set.jsonl", encoding="utf-8") if l.strip()]
    return items


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=config.DEFAULT_PROFILE, choices=list(config.PROFILES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    items = load_set()
    if args.limit:
        items = items[: args.limit]
    prof = config.PROFILES[args.profile]

    # 앞 조건의 모델이 메모리에 남아 있으면 숫자가 섞인다 → 전부 내린다.
    for m in ollama.loaded():
        ollama.unload(m["name"])
    # 모델 프로세스(llama-server)가 실제로 사라질 때까지 기다린다. 남아 있으면 메모리가 부풀려진다.
    for _ in range(60):
        if not any((p.info["name"] or "").lower().startswith("llama-server")
                   for p in psutil.process_iter(["name"])):
            break
        time.sleep(1)
    time.sleep(2)
    vram_base = MemoryMonitor.vram_used_mib()

    eng = Engine(args.profile)
    mon = MemoryMonitor()
    mon.start()
    # 데우기: 첫 질문은 모델을 디스크에서 메모리로 올리는 시간이 섞인다 → 따로 재고 응답 시간에선 뺀다.
    t0 = time.perf_counter()
    eng.ask("이 규정의 목적은 무엇인가요?")
    load_s = time.perf_counter() - t0
    # 모델을 올리는 순간에는 Ollama가 임시 프로세스를 띄워 메모리가 잠깐 크게 튄다.
    # 그래서 «올리는 중 최대»(mon)와 «질문에 답하는 중 최대»(mon_q)를 따로 잰다.
    time.sleep(3)
    mon_q = MemoryMonitor()
    mon_q.start()
    t_start = time.perf_counter()
    runs_dir = HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    name = args.profile + (f"-{args.tag}" if args.tag else "")
    out_f = open(runs_dir / f"{name}.jsonl", "w", encoding="utf-8")
    rows = []
    for i, it in enumerate(items, 1):
        a = eng.ask(it["question"])
        gold = [base_id(g) for g in it.get("gold", [])]
        cited = [base_id(c) for c in a.citations]
        retrieved = [base_id(c["chunk_id"]) for c in a.contexts]
        row = {
            "id": it["id"], "type": it["type"], "question": it["question"],
            "gold": gold, "keywords": it.get("keywords", []),
            "pred_answerable": a.answerable, "pred_answer": a.answer, "cited": cited,
            "retrieved": retrieved, "timing": a.timing,
        }
        if it["type"] == "answerable":
            row["correct"] = a.answerable and all(norm(k) in norm(a.answer) for k in it["keywords"])
            row["cite_ok"] = any(g in cited for g in gold)
            row["retrieved_ok"] = any(g in retrieved for g in gold)
        else:
            row["correct"] = not a.answerable
        rows.append(row)
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        mark = "O" if row["correct"] else "X"
        print(f"[{i:3}/{len(items)}] {mark} {a.timing['total_s']:5.1f}s {it['id']} {it['question'][:50]}", flush=True)
    wall = time.perf_counter() - t_start
    for m_ in (mon, mon_q):
        m_.stop_flag.set()
        m_.join()

    ans = [r for r in rows if r["type"] == "answerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    times = sorted(r["timing"]["total_s"] for r in rows)
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
    summary = {
        "profile": name, "model": prof["model"], "options": prof["options"],
        "embed_model": eng.embed_model, "n_answerable": len(ans), "n_unanswerable": len(una),
        "accuracy": pct([r["correct"] for r in ans]),
        "citation_match": pct([r["cite_ok"] for r in ans]),
        "unknown_correct": pct([r["correct"] for r in una]),
        "false_refusal": pct([not r["pred_answerable"] for r in ans]),
        "retrieval_recall@5": pct([r["retrieved_ok"] for r in ans]),
        "latency_mean_s": round(statistics.mean(times), 2),
        "latency_p50_s": round(statistics.median(times), 2),
        "latency_p95_s": round(times[int(0.95 * (len(times) - 1))], 2),
        "peak_ram_gb": round(mon_q.peak_ram / 1024**3, 2),          # 질문에 답하는 동안
        "peak_vram_gb": round(mon_q.peak_vram / 1024, 2),
        "peak_ram_incl_load_gb": round(mon.peak_ram / 1024**3, 2),  # 모델 올리는 순간 포함
        "peak_vram_incl_load_gb": round(mon.peak_vram / 1024, 2),
        "vram_baseline_gb": round(vram_base / 1024, 2),
        "warmup_s": round(load_s, 1),
        "peak_ram_by_process_gb": {k: round(v / 1024**3, 2) for k, v in mon.by_proc.items()},
        "wall_s": round(wall, 1),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    (runs_dir / f"{name}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
