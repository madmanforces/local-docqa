"""평가 세트 자동 검증 — 법 지식 없이 기계로 확인할 수 있는 것은 모두 기계로 확인한다.

build_set check(만들 때 하는 검사)에 더해, 여기서는 다음을 본다.
  [답 있는 85문항]
  1) 핵심어가 들어 있는 문단(항)을 찾는다.
  2) 그 문단이 질문과 같은 이야기인지: 질문 글자쌍(2글자 묶음)이 그 문단에 얼마나 겹치는지(겹침률).
     핵심어가 조문 안 엉뚱한 문단에만 있으면 겹침률이 낮게 나온다.
  3) 네 조건(모델 3종) 모두가 틀린 문항: 문제 쪽이 잘못됐을 수 있다.
  [근거 없는 15문항]
  4) 확인 낱말(check_terms)이 그 문서뿐 아니라 8개 문서 전체 어디에도 없는가.
겹침률이 낮거나 모두 틀린 문항만 사람 확인용 시트(review_sample.md)에 올린다.

사용: uv run python -m evaluation.verify_set
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from evaluation.build_set import load_docs, norm

HERE = Path(__file__).resolve().parent
RUNS = ["gpu", "gpu-qwen2.5", "cpu", "cpu-4t"]
LOW = 0.35   # 겹침률이 이보다 낮으면 사람 확인으로 보낸다
GENERIC = ["무엇인가요", "어떻게", "되나요", "하나요", "인가요", "있나요", "경우", "얼마", "며칠", "몇"]


def bigrams(s: str) -> set[str]:
    s = norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def strip_question(q: str, title: str) -> str:
    q = q.split("에서", 1)[1] if "에서" in q[:40] else q       # 「○○ 규정에서」 머리말 제거
    q = re.sub(r"^\s*따르면", "", q)
    for g in GENERIC:
        q = q.replace(g, " ")
    return q


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    docs = load_docs()
    by_id = {c.chunk_id: c for _, chunks in docs.values() for c in chunks}
    corpus = norm(" ".join(c.text for _, chunks in docs.values() for c in chunks))
    items = [json.loads(l) for l in open(HERE / "qa_set.jsonl", encoding="utf-8") if l.strip()]
    results = {r: {json.loads(l)["id"]: json.loads(l)["correct"]
                   for l in open(HERE / "runs" / f"{r}.jsonl", encoding="utf-8")} for r in RUNS}

    rows, flagged = [], []
    for it in items:
        if it["type"] == "unanswerable":
            hits = [t for t in it["check_terms"] if norm(t) in corpus]
            rows.append({"id": it["id"], "type": "unanswerable", "terms": it["check_terms"],
                         "found_anywhere": hits, "ok": not hits})
            continue
        title = docs[it["doc_id"]][0]
        qb = bigrams(strip_question(it["question"], title))
        paras = [p for g in it["gold"] for p in by_id[g].text.split("\n") if p.strip()]
        best, best_p = 0.0, ""
        for p in paras:
            if all(norm(k) in norm(p) for k in it["keywords"]):
                sc = len(qb & bigrams(p)) / max(1, len(qb))
                if sc > best:
                    best, best_p = sc, p
        if not best_p:   # 핵심어가 한 문단에 다 모여 있지 않음 → 문단마다 따로
            best_p = " / ".join(p for p in paras if any(norm(k) in norm(p) for k in it["keywords"]))
        all_wrong = not any(results[r][it["id"]] for r in RUNS)
        reasons = []
        if best < LOW:
            reasons.append(f"질문과 문단 겹침률 낮음({best:.2f})")
        if all_wrong:
            reasons.append("네 조건 모두 틀림")
        row = {"id": it["id"], "type": "answerable", "overlap": round(best, 2),
               "all_runs_wrong": all_wrong, "flag": reasons, "para": best_p}
        rows.append(row)
        if reasons:
            flagged.append((it, row))

    (HERE / "verify_report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    ans = [r for r in rows if r["type"] == "answerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    print(f"답 있는 문항 {len(ans)}개: 사람 확인으로 보냄 {len(flagged)}개")
    for it, r in flagged:
        print(f"  {r['id']} {r['flag']}")
    print(f"근거 없는 문항 {len(una)}개: 8개 문서 전체에서 확인 낱말 없음 {sum(r['ok'] for r in una)}개")
    for r in una:
        if not r["ok"]:
            print(f"  {r['id']} 다른 문서에 있음: {r['found_anywhere']}")
    return flagged


if __name__ == "__main__":
    main()
