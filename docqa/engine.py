"""질문 → 검색 → LLM 답변(근거 인용) 흐름.

환각을 줄이는 장치 세 가지:
1. 검색된 조문만 보여 주고 «이 안에서만 답하라»고 지시한다.
2. 답을 JSON 틀(schema)로 받는다: 답할 수 있는지(answerable), 답, 인용 번호.
   → 사람이 읽는 글이 아니라 기계가 검사할 수 있는 모양이라 근거 일치를 자동으로 잴 수 있다.
3. 인용 번호가 실제로 보여 준 조문 번호가 아니면 버린다(지어낸 근거 차단).
   답할 수 있다고 했는데 남는 인용이 하나도 없으면 «모른다»로 바꾼다.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import config, ollama
from .store import Store

UNKNOWN = "제공된 규정에서 근거를 찾을 수 없어 답할 수 없습니다."

SYSTEM = """당신은 회사 규정 문서 질의응답 도우미입니다.
규칙:
- 반드시 아래 [조문] 안의 내용만 근거로 답합니다. 일반 상식이나 다른 법령 지식으로 채우지 않습니다.
- 질문에 답하는 내용이 [조문]에 분명히 없으면 answerable을 false로 하고 answer는 비워 둡니다.
  질문이 묻는 문서·기관과 조문의 문서·기관이 다르면 그것도 근거가 없는 것입니다.
- 답할 수 있으면 answer에 한두 문장으로 핵심(숫자·기간·주체 등)을 그대로 적고,
  citations에 근거가 된 조문 번호(예: 2)를 넣습니다."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "answer", "citations"],
}


@dataclass
class Answer:
    question: str
    answerable: bool
    answer: str
    citations: list[str]                     # chunk_id 목록 (예: «01:제31조»)
    contexts: list[dict] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    raw: str = ""

    def pretty(self, titles: dict[str, str]) -> str:
        if not self.answerable:
            return UNKNOWN
        lines = [self.answer, "", "근거:"]
        by_id = {c["chunk_id"]: c for c in self.contexts}
        for cid in self.citations:
            c = by_id[cid]
            lines.append(f"- {titles.get(c['doc_id'], c['doc_id'])} {c['article']}({c['title']})")
            lines.append("  " + c["text"][:300].replace("\n", "\n  "))
        return "\n".join(lines)


class Engine:
    def __init__(self, profile: str = config.DEFAULT_PROFILE, store: Store | None = None):
        self.profile = config.PROFILES[profile]
        self.store = store or Store(config.DB_PATH)
        self.titles = self.store.doc_titles()
        self.embed_model = self.store.get_meta("embed_model", config.EMBED_MODEL)

    def retrieve(self, question: str, k: int = config.TOP_K) -> list[dict]:
        qvec = ollama.embed(self.embed_model, [question], self.profile.get("embed_options"))[0]
        return self.store.hybrid_search(question, qvec, k=k)

    def build_prompt(self, question: str, contexts: list[dict]) -> str:
        blocks = []
        for i, c in enumerate(contexts, 1):
            blocks.append(f"[{i}] {self.titles.get(c['doc_id'], c['doc_id'])} {c['article']}\n{c['text']}")
        return "[조문]\n" + "\n\n".join(blocks) + f"\n\n[질문]\n{question}"

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        contexts = self.retrieve(question)
        t1 = time.perf_counter()
        res = ollama.chat(
            self.profile["model"],
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": self.build_prompt(question, contexts)}],
            schema=SCHEMA, options=self.profile["options"],
        )
        t2 = time.perf_counter()
        raw = res["message"]["content"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"answerable": False, "answer": "", "citations": []}
        cites = []
        for n in data.get("citations") or []:
            if isinstance(n, int) and 1 <= n <= len(contexts):
                cid = contexts[n - 1]["chunk_id"]
                if cid not in cites:
                    cites.append(cid)
        answerable = bool(data.get("answerable")) and bool(cites) and bool((data.get("answer") or "").strip())
        return Answer(
            question=question,
            answerable=answerable,
            answer=(data.get("answer") or "").strip() if answerable else "",
            citations=cites if answerable else [],
            contexts=contexts,
            timing={"retrieve_s": t1 - t0, "generate_s": t2 - t1, "total_s": t2 - t0,
                    "eval_tokens": res.get("eval_count"), "prompt_tokens": res.get("prompt_eval_count")},
            raw=raw,
        )
