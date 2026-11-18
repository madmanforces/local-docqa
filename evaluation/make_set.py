"""평가 세트 후보를 자동으로 만든다 (답이 있는 질문).

방법:
1. 문서마다 조문을 무작위로 고른다 (짧은 조문·«삭제» 조문은 뺀다). seed 고정 → 다시 돌려도 같은 후보.
2. 로컬 LLM(qwen2.5:7b, GPU)에게 조문 하나를 주고 «질문 / 짧은 정답 / 정답 핵심어»를 JSON으로 받는다.
3. 기계 검사로 거른다:
   - 핵심어가 조문 원문에 글자 그대로 있어야 한다 (지어낸 정답 차단).
   - 핵심어가 질문 안에 이미 들어 있으면 안 된다 (질문만 보고 맞히는 문제 차단).
   - 질문에 문서 이름(기관명 또는 규정명)이 들어 있어야 한다 (문서 8개라 «어느 규정인지» 필요).
4. 남은 후보는 사람이 검토해 qa_set.jsonl로 옮긴다 (evaluation/review.py 참고).

사용: uv run python -m evaluation.make_set --per-doc 18 --out evaluation/candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

from docqa import config, ollama
from docqa.cli import doc_title
from docqa.hwpx import read_paragraphs
from docqa.structure import split_articles

# 생성 모델은 평가 대상(qwen3.5, qwen2.5)과 다른 모델을 쓴다 — 자기가 만든 문제를 자기가 푸는 편향을 피하려고.
# (처음엔 qwen2.5:7b로 해 봤으나 중국어가 섞이고 핵심어가 원문에 없는 경우가 많아 바꿨다.)
GEN_MODEL = "qwen3-vl:8b"

PROMPT = """다음은 「{title}」의 한 조문입니다.

{text}

이 조문만 읽으면 답할 수 있는 질문 하나를 만드세요.
조건:
- 직원이 실제로 물어볼 법한 자연스러운 한국어 질문. 문서 이름은 쓰지 않습니다(나중에 붙입니다).
- 조 번호(제○조)는 질문에 쓰지 않습니다.
- 정답은 숫자·기간·주체·조건처럼 짧고 분명한 사실이어야 합니다.
- keywords: 정답의 핵심이 되는 짧은 낱말 1~3개. 조문에 글자 그대로 있는 말만 씁니다. 질문에 이미 나온 말은 쓰지 않습니다.
JSON으로 답하세요: {{"question": "...", "answer": "...", "keywords": ["..."]}}"""

SCHEMA = {"type": "object",
          "properties": {"question": {"type": "string"}, "answer": {"type": "string"},
                         "keywords": {"type": "array", "items": {"type": "string"}}},
          "required": ["question", "answer", "keywords"]}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def fix(item: dict, text: str, title: str) -> dict:
    """기계적으로 고칠 수 있는 것은 고친다: 문서 이름 붙이기, 원문에 없거나 질문에 있는 핵심어 빼기."""
    q = item.get("question", "").strip()
    short = re.sub(r"\(([^)]*)\)\s*", r"\1 ", title).strip()      # «건강보험심사평가원 인사규정»
    if norm(re.sub(r"\([^)]*\)", "", title)) not in norm(q):
        q = f"{short}에서 {q}"
    kws = [k.strip() for k in item.get("keywords", []) if k.strip()]
    kws = [k for k in kws if norm(k) in norm(text) and norm(k) not in norm(q)]
    return {**item, "question": q, "keywords": kws}


def check(item: dict, text: str, title: str) -> str | None:
    q, kws = item.get("question", ""), item.get("keywords", [])
    if re.search(r"[一-鿿぀-ヿ]", q + item.get("answer", "")):
        return "한자·일본어 섞임"
    if not q or not kws:
        return "남는 핵심어 없음"
    if re.search(r"제\s*\d+\s*조", q):
        return "질문에 조 번호"
    for k in kws:
        if norm(k) not in norm(text):
            return f"핵심어가 원문에 없음: {k}"
        if norm(k) in norm(q):
            return f"핵심어가 질문에 있음: {k}"
    org = re.search(r"\(([^)]+)\)", title)
    names = [norm(re.sub(r"\([^)]*\)", "", title))] + ([norm(org.group(1))] if org else [])
    if not any(n and n in norm(q) for n in names):
        return "질문에 문서 이름 없음"
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=18)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="evaluation/candidates.jsonl")
    args = ap.parse_args()

    # 색인(DB) 없이 원본 HWPX에서 바로 조문을 읽는다 — 평가 세트가 검색 엔진에 기대지 않도록.
    docs = {}
    for fp in sorted((config.DATA / "raw").glob("*.hwpx")):
        paras = read_paragraphs(fp)
        docs[fp.stem] = (doc_title(paras, fp.stem), split_articles(fp.stem, paras))
    titles = {k: v[0] for k, v in docs.items()}
    rng = random.Random(args.seed)
    kept = dropped = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for doc_id in sorted(titles):
            rows = [(c.chunk_id, c.article, c.title, c.text) for c in docs[doc_id][1]]
            pool = [r for r in rows if len(r[3]) >= 80 and "삭제" not in r[3][:30] and r[2] not in ("목적",)]
            rng.shuffle(pool)
            for cid, article, atitle, text in pool[: args.per_doc]:
                title = titles[doc_id]
                short = re.sub(r"[()]", "", title)
                res = ollama.chat(GEN_MODEL, [{"role": "user", "content": PROMPT.format(
                    title=title, text=text, short=short)}], schema=SCHEMA,
                    options={"temperature": 0.3, "num_ctx": 4096, "seed": args.seed}, think=False)
                try:
                    item = json.loads(res["message"]["content"])
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                item = fix(item, text, title)
                why = check(item, text, title)
                rec = {"doc_id": doc_id, "gold": [cid], "article_title": atitle, **item,
                       "auto_check": why or "ok"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += why is None
                dropped += why is not None
                print(flush=True, *[f"{cid:14} {'OK ' if why is None else 'X  '} {item.get('question', '')[:70]}  {why or ''}"])
    print(f"통과 {kept}, 탈락 {dropped} → {args.out}")


if __name__ == "__main__":
    main()
