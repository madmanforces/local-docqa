"""사람 확인용 시트 — 법 지식 없이 «글자 맞추기»만 하면 되는 꼴.

기계 검증(verify_set.py)에서 «핵심어가 조문 안 다른 줄(목록 항목)에 떨어져 있는» 문항 중 5개를 골랐다.
각 문항은 원문 몇 줄과 «찾을 글자» 두 개를 보여 주고, 두 글자가 주어진 줄에서 서로 이어지는지만 묻는다.

사용: uv run python -m evaluation.review_sheet  → evaluation/review_sample.md
"""
import json
import sys
from pathlib import Path

from evaluation.build_set import load_docs, norm

HERE = Path(__file__).resolve().parent

# (문항 id, 질문에서 찾을 글자, 정답 글자, 보여 줄 줄을 고르는 낱말들)
PICKS = [
    ("a007", "6급나", "3년", ["근속승진", "6급나"]),
    ("a012", "남성직원", "1일", ["난임치료시술휴가", "정자 채취일"]),
    ("a055", "자산가액", "취득가액", ["계산의 원칙", "자산가액"]),
    ("a075", "주요사업의 합법성 검토", "이사장", ["직무배분", "이사장의 결재사항", "주요사업의 합법성", "본부장의 전결사항"]),
    ("a081", "위원장", "부원장", ["위원장은"]),
]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    docs = load_docs()
    by_id = {c.chunk_id: c for _, chunks in docs.values() for c in chunks}
    items = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "qa_set.jsonl", encoding="utf-8")}
    out = ["""# 평가 세트 확인 (5문항, 글자 맞추기)

법 지식은 필요 없습니다. 문항마다 아래 원문 줄을 보고,
**«질문 글자»와 «정답 글자»가 같은 줄에 있거나, 위 제목 줄 → 아래 항목으로 이어지는지**만 봐 주세요.

- 이어지면 `[O]`, 아니면 `[X]`, 모르겠으면 `[?]` 로 고쳐 적어 주세요.
- 원문은 필요한 줄만 보여 줍니다(사이 줄은 생략).
- 나머지 80문항과 근거 없는 15문항은 기계로 확인했습니다(`evaluation/verify_set.py`).
"""]
    for n, (qid, qword, aword, anchors) in enumerate(PICKS, 1):
        it = items[qid]
        # 긴 조는 조각(#2 등)을 모두 본다 — 제목 줄이 앞 조각에 있을 수 있다.
        bases = {g.split("#")[0] for g in it["gold"]}
        lines = [l for cid, c in by_id.items() if cid.split("#")[0] in bases for l in c.text.split("\n")]
        shown = []
        for l in lines:
            if any(norm(a) in norm(l) for a in anchors) and l not in shown:
                shown.append(l if len(l) <= 160 else l[:160] + " …")
        out.append(f"\n## {n}. [ ] {qid}\n\n- 질문: {it['question']}\n- 질문 글자: **{qword}**\n- 정답 글자: **{aword}**\n- 원문 ({', '.join(it['gold'])}):\n\n")
        out.append("\n>\n".join(f"> {l}" for l in shown) + "\n")
    (HERE / "review_sample.md").write_text("".join(out), encoding="utf-8")
    print("→ evaluation/review_sample.md")
