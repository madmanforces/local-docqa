"""사람 검토용 표본 시트를 만든다: 답 있는 문항에서 20개를 무작위로 뽑아 근거 조문과 나란히 보여 준다.

사용: uv run python -m evaluation.review_sheet  → evaluation/review_sample.md
검토자는 각 문항의 [ ]에 O(맞음)/X(틀림)/?(애매)를 적는다.
"""
import json
import random
import sys
from pathlib import Path

from evaluation.build_set import load_docs

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    docs = load_docs()
    by_id = {c.chunk_id: c for _, chunks in docs.values() for c in chunks}
    items = [json.loads(l) for l in open(HERE / "qa_set.jsonl", encoding="utf-8")]
    ans = [i for i in items if i["type"] == "answerable"]
    sample = sorted(random.Random(2026).sample(ans, 20), key=lambda x: x["id"])
    out = ["# 평가 세트 표본 검토 (20/85, seed=2026)\n\n각 문항: 정답이 근거 조문과 맞으면 O, 틀리면 X, 애매하면 ?\n"]
    for it in sample:
        out.append(f"\n## [ ] {it['id']} — {it['question']}\n\n- 정답: {it['answer']}\n- 핵심어: {', '.join(it['keywords'])}\n- 근거: {', '.join(it['gold'])}\n")
        for g in it["gold"]:
            out.append("\n> " + by_id[g].text.replace("\n", "\n> ") + "\n")
    (HERE / "review_sample.md").write_text("".join(out), encoding="utf-8")
    print("→ evaluation/review_sample.md")
