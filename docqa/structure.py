"""문단 목록 → 검색 단위(조 단위 덩어리) 목록.

왜 «조» 단위인가:
- 법령 질문의 근거는 보통 «제○조»로 말한다. 정답 근거와 비교하기 쉽다.
- 항(①②) 하나만 떼면 주어·조건이 앞 항에 있어 뜻이 끊기는 일이 많다.
- 조가 너무 길면(MAX_CHARS 넘음) 항 경계에서 나눠 «제○조 (2)»처럼 이어 붙인다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
RE_PART = re.compile(r"^제\s*\d+\s*(편|장|절|관)(\s|$)")
RE_JO = re.compile(r"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(?:\(([^)]*)\))?")
RE_BUCHIK = re.compile(r"^부\s*칙")
RE_BYEOL = re.compile(r"^[\[<【]?\s*별\s*(표|지)\s*(\d*)")

MAX_CHARS = 1500


@dataclass
class Chunk:
    doc_id: str      # «01»
    article: str     # «제14조», «부칙1/제2조», «별표1»
    title: str       # 조 제목 «(휴직)»의 안쪽
    text: str        # 본문 (조 머리 포함)
    part: int = 1    # 긴 조를 나눴을 때 몇 번째 조각인지

    @property
    def chunk_id(self) -> str:
        base = f"{self.doc_id}:{self.article}"
        return base if self.part == 1 else f"{base}#{self.part}"


def _ws(s: str) -> str:
    return re.sub(r"[ \t　]+", " ", s).strip()


def _jo_key(s: str):
    """문단이 조 머리면 (열쇠, 제목), 아니면 None."""
    m = RE_JO.match(s)
    if not m:
        return None
    # «제3조에 따라 ...» 같은 본문 속 인용을 조 머리로 착각하지 않도록:
    # 괄호 제목이 있거나, «제3조 삭제»처럼 끝나는 경우만 조 머리로 본다.
    if m.group(3) is None and not re.match(r"^제\s*\d+\s*조(의\s*\d+)?\s*(삭제|$)", s):
        return None
    key = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
    return key, (m.group(3) or "").strip()


def split_articles(doc_id: str, paragraphs: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""          # «» 본칙, «부칙1», «부칙2» ...
    n_buchik = 0
    cur: dict | None = None
    started = False

    def flush():
        if cur and cur["lines"]:
            chunks.extend(_split_long(doc_id, cur["key"], cur["title"], cur["lines"]))

    for raw in paragraphs:
        s = _ws(raw)
        if not s:
            continue
        jo = _jo_key(s)
        if not started:   # 제목·연락처 같은 앞부분은 건너뛴다
            if jo or RE_PART.match(s):
                started = True
            else:
                continue
        if RE_BUCHIK.match(s) and len(s) < 60:
            flush()
            n_buchik += 1
            section = f"부칙{n_buchik}"
            cur = {"key": section, "title": "부칙", "lines": [s]}
            continue
        m = RE_BYEOL.match(s)
        if m and len(s) < 200 or (m and section.startswith("별표")):
            flush()
            section = "별표"
            key = f"별{m.group(1)}{m.group(2) or ''}"
            cur = {"key": key, "title": s[:40], "lines": [s]}
            continue
        if RE_PART.match(s) and len(s) < 60:   # «제2장 인사위원회»는 단위가 아니라 구분선
            continue
        if jo:
            flush()
            key = (section + "/" if section and not section.startswith("별표") else "") + jo[0]
            cur = {"key": key, "title": jo[1], "lines": [s]}
            continue
        if cur is not None:
            cur["lines"].append(s)
    flush()
    return _dedupe(chunks)


def _split_long(doc_id: str, key: str, title: str, lines: list[str]) -> list[Chunk]:
    text = "\n".join(lines)
    if len(text) <= MAX_CHARS:
        return [Chunk(doc_id, key, title, text)]
    # 항(①) 또는 호(1.) 경계에서 MAX_CHARS 이하로 묶는다. 첫 줄(조 머리)은 조각마다 앞에 붙인다.
    head = lines[0][:80]
    out, buf, part = [], [], 1
    for ln in lines:
        if buf and sum(len(x) for x in buf) + len(ln) > MAX_CHARS:
            body = "\n".join(buf)
            out.append(Chunk(doc_id, key, title, body if part == 1 else f"{head} …(이어서)\n{body}", part))
            buf, part = [], part + 1
        buf.append(ln[: MAX_CHARS * 2])
    if buf:
        body = "\n".join(buf)
        out.append(Chunk(doc_id, key, title, body if part == 1 else f"{head} …(이어서)\n{body}", part))
    return out


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    """같은 열쇠가 두 번 나오면(예: 별표 번호 없음) 뒤에 «~2»를 붙여 구분한다."""
    seen: dict[str, int] = {}
    for c in chunks:
        k = c.chunk_id
        if k in seen:
            seen[k] += 1
            c.article = f"{c.article}~{seen[k]}"
        else:
            seen[k] = 1
    return chunks
