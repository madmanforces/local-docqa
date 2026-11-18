"""명령줄 사용법:
  uv run docqa ingest                 # data/raw/*.hwpx → data/index.sqlite
  uv run docqa ask "질문" [--profile cpu]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import config, ollama
from .hwpx import read_paragraphs
from .store import Store
from .structure import split_articles


def doc_title(paragraphs: list[str], fallback: str) -> str:
    for p in paragraphs[:8]:
        s = p.strip()
        if s and not s.startswith("⟨") and not s.startswith("["):
            return re.sub(r"\s+", " ", s)
    return fallback


def ingest(raw_dir: Path, db_path: Path, embed_model: str, batch: int = 16) -> None:
    store = Store(db_path)
    store.reset()
    files = sorted(raw_dir.glob("*.hwpx"))
    all_chunks = []
    for f in files:
        paras = read_paragraphs(f)
        title = doc_title(paras, f.stem)
        store.add_doc(f.stem, title, f.name)
        chunks = split_articles(f.stem, paras)
        all_chunks += chunks
        print(f"  {f.name}: {title} — {len(chunks)}개 조문 덩어리")
    t0 = time.perf_counter()
    for i in range(0, len(all_chunks), batch):
        part = all_chunks[i:i + batch]
        # 임베딩할 글 앞에 조 제목을 붙이면 «무엇에 관한 조문인지»가 벡터에 더 잘 담긴다.
        vecs = ollama.embed(embed_model, [f"{c.article}({c.title}) {c.text}" for c in part])
        for c, v in zip(part, vecs):
            store.add_chunk(c.chunk_id, c.doc_id, c.article, c.title, c.text, v)
    store.set_meta("embed_model", embed_model)
    store.commit()
    print(f"완료: 문서 {len(files)}개, 덩어리 {len(all_chunks)}개, 임베딩 {time.perf_counter() - t0:.1f}초 → {db_path}")


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="docqa", description="폐쇄망 문서 Q&A (로컬 LLM)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="HWPX를 읽어 색인을 만든다")
    p.add_argument("--raw", type=Path, default=config.DATA / "raw")
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--embed-model", default=config.EMBED_MODEL)
    q = sub.add_parser("ask", help="질문한다")
    q.add_argument("question")
    q.add_argument("--profile", default=config.DEFAULT_PROFILE, choices=list(config.PROFILES))
    q.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    args = ap.parse_args(argv)

    if args.cmd == "ingest":
        ingest(args.raw, args.db, args.embed_model)
    else:
        from .engine import Engine
        eng = Engine(args.profile)
        a = eng.ask(args.question)
        if args.json:
            print(json.dumps({"answerable": a.answerable, "answer": a.answer, "citations": a.citations,
                              "timing": a.timing}, ensure_ascii=False, indent=2))
        else:
            print(a.pretty(eng.titles))
            print(f"\n({a.timing['total_s']:.1f}초, 모델 {eng.profile['model']})")


if __name__ == "__main__":
    main()
