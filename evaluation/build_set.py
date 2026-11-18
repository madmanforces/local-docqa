"""평가 세트 만들기 (2단계).

  1) sample : 문서마다 조문을 무작위로 뽑아 sampled_articles.md로 쓴다 (seed 고정).
  2) check  : qa_draft.jsonl(뽑힌 조문으로 쓴 질문 초안) + unanswerable.jsonl을 기계 검사한 뒤
              qa_set.jsonl로 합친다.

기계 검사 (하나라도 걸리면 실패로 멈춘다):
  - 정답 근거(gold) 조문이 실제로 있는가
  - 정답 핵심어(keywords)가 근거 조문 원문에 글자 그대로 있는가 (띄어쓰기 무시)
  - 핵심어가 질문에 이미 들어 있지 않은가 (질문만 보고 맞히는 문제 방지)
  - 질문에 조 번호(제○조)가 없는가, 질문에 문서 이름이 들어 있는가
  - 근거 없는 질문: check_terms 낱말이 그 문서 원문 어디에도 없는가

사용:
  uv run python -m evaluation.build_set sample
  uv run python -m evaluation.build_set check
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from docqa import config
from docqa.cli import doc_title
from docqa.hwpx import read_paragraphs
from docqa.structure import split_articles

HERE = Path(__file__).resolve().parent
SEED = 7
# 문서별 문항 수: 조문 수에 비례하되 작은 문서도 최소 5개. 합계 85 (+ 근거 없는 질문 15 = 100).
QUOTA = {"01": 14, "02": 8, "03": 10, "04": 8, "05": 18, "06": 13, "07": 5, "08": 9}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def load_docs():
    docs = {}
    for fp in sorted((config.DATA / "raw").glob("*.hwpx")):
        paras = read_paragraphs(fp)
        docs[fp.stem] = (doc_title(paras, fp.stem), split_articles(fp.stem, paras))
    return docs


def sample() -> None:
    docs = load_docs()
    rng = random.Random(SEED)
    out = ["# 평가용으로 뽑힌 조문 (seed=7, 문서별 할당량의 1.5배를 뽑아 부적합한 것은 건너뜀)\n"]
    for doc_id, (title, chunks) in docs.items():
        pool = [c for c in chunks if len(c.text) >= 80 and not re.match(r"^제\s*\d+\s*조(의\s*\d+)?\s*(\([^)]*\))?\s*삭제", c.text)
                and c.title not in ("목적",)]
        rng.shuffle(pool)
        out.append(f"\n## {doc_id} {title}\n")
        for c in pool[: int(QUOTA[doc_id] * 1.5) + 1]:
            out.append(f"\n### {c.chunk_id} ({c.title})\n\n{c.text}\n")
    (HERE / "sampled_articles.md").write_text("".join(out), encoding="utf-8")
    print("→ evaluation/sampled_articles.md")


def check() -> None:
    docs = load_docs()
    by_id = {c.chunk_id: c for _, chunks in docs.values() for c in chunks}
    full = {d: norm(" ".join(c.text for c in chunks)) for d, (_, chunks) in docs.items()}
    errors, items = [], []
    drafts = [json.loads(l) for l in open(HERE / "qa_draft.jsonl", encoding="utf-8") if l.strip()]
    for n, it in enumerate(drafts, 1):
        q, kws = it["question"], it["keywords"]
        title = docs[it["doc_id"]][0]
        org = re.search(r"\(([^)]*)\)", title).group(1)
        for g in it["gold"]:
            if g not in by_id:
                errors.append(f"{n}: 근거 조문 없음 {g}")
        gold_text = norm(" ".join(by_id[g].text for g in it["gold"] if g in by_id))
        for k in kws:
            if norm(k) not in gold_text:
                errors.append(f"{n}: 핵심어가 원문에 없음 «{k}» ({it['gold']})")
            if norm(k) in norm(q):
                errors.append(f"{n}: 핵심어가 질문에 있음 «{k}»")
        if re.search(r"제\s*\d+\s*조", q):
            errors.append(f"{n}: 질문에 조 번호")
        if norm(org) not in norm(q):
            errors.append(f"{n}: 질문에 기관 이름 없음")
        items.append({"id": f"a{n:03}", "type": "answerable", **it})
    for it in (json.loads(l) for l in open(HERE / "unanswerable.jsonl", encoding="utf-8") if l.strip()):
        for t in it["check_terms"]:
            if norm(t) in full[it["doc_id"]]:
                errors.append(f"{it['id']}: «{t}»가 문서에 있음 — 근거 없는 질문이 아닐 수 있음")
        items.append({"type": "unanswerable", "gold": [], "keywords": [], "answer": "", **it})
    if errors:
        print("\n".join(errors))
        sys.exit(f"검사 실패 {len(errors)}건")
    with open(HERE / "qa_set.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    na = sum(i["type"] == "answerable" for i in items)
    print(f"검사 통과 → qa_set.jsonl: 답 있는 질문 {na}개, 근거 없는 질문 {len(items) - na}개")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"sample": sample, "check": check}[sys.argv[1]]()
