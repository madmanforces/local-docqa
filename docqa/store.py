"""SQLite 하나에 조문과 벡터를 함께 저장하고 검색한다.

왜 벡터 DB(서버)를 따로 안 쓰나:
- 폐쇄망 PC에 설치할 것이 적을수록 좋다. SQLite는 파이썬에 들어 있고 파일 하나다.
- 조문 수가 수천 개 수준이면 전부 꺼내 numpy로 코사인 유사도를 계산해도 수 ms면 끝난다.
  (수십만 개가 되면 sqlite-vec 같은 확장이나 ANN 색인으로 바꾸면 된다.)

검색은 두 가지를 섞는다(하이브리드):
- 벡터 검색: 뜻이 비슷한 조문 (예: «쉬는 것» ↔ «휴직»)
- 글자 검색: SQLite FTS5 trigram — 「제14조」, 「60일」 같은 정확한 낱말
두 순위를 RRF(순위 역수 합)로 합친다.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(doc_id TEXT PRIMARY KEY, title TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY, doc_id TEXT, article TEXT, title TEXT, text TEXT, vec BLOB);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='trigram');
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self._cache = None

    # ---------- 쓰기 ----------
    def reset(self):
        self.db.executescript("DELETE FROM docs; DELETE FROM chunks; DELETE FROM chunks_fts; DELETE FROM meta;")
        self._cache = None

    def set_meta(self, key: str, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def add_doc(self, doc_id: str, title: str, source: str):
        self.db.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?)", (doc_id, title, source))

    def add_chunk(self, chunk_id, doc_id, article, title, text, vec):
        v = np.asarray(vec, dtype=np.float32)
        v /= np.linalg.norm(v) or 1.0          # 미리 길이 1로 맞춰 두면 내적 = 코사인
        self.db.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?,?)",
                        (chunk_id, doc_id, article, title, text, v.tobytes()))
        self.db.execute("INSERT INTO chunks_fts VALUES(?,?)", (chunk_id, text))

    def commit(self):
        self.db.commit()
        self._cache = None

    # ---------- 읽기 ----------
    def doc_titles(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT doc_id, title FROM docs"))

    def get(self, chunk_id: str) -> dict | None:
        row = self.db.execute("SELECT chunk_id, doc_id, article, title, text FROM chunks WHERE chunk_id=?",
                              (chunk_id,)).fetchone()
        return dict(zip(("chunk_id", "doc_id", "article", "title", "text"), row)) if row else None

    def _load(self):
        if self._cache is None:
            rows = self.db.execute("SELECT chunk_id, vec FROM chunks ORDER BY rowid").fetchall()
            ids = [r[0] for r in rows]
            mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1))
            self._cache = (ids, mat)
        return self._cache

    def vector_search(self, qvec, k: int = 20) -> list[tuple[str, float]]:
        ids, mat = self._load()
        q = np.asarray(qvec, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        scores = mat @ q
        top = np.argsort(-scores)[:k]
        return [(ids[i], float(scores[i])) for i in top]

    def text_search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        # trigram은 3글자 이상 낱말만 찾는다. 질문을 낱말로 쪼개 OR로 묶는다.
        words = [w for w in re.findall(r"[\w]+", query) if len(w) >= 3]
        if not words:
            return []
        q = " OR ".join('"' + w.replace('"', "") + '"' for w in words[:12])
        try:
            rows = self.db.execute(
                "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY 2 LIMIT ?",
                (q, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r[0], -float(r[1])) for r in rows]

    def hybrid_search(self, query: str, qvec, k: int = 5, rrf_k: int = 60) -> list[dict]:
        """벡터 순위와 글자 순위를 RRF로 합친다: 점수 = Σ 1/(60 + 순위)."""
        fused: dict[str, float] = {}
        for ranking in (self.vector_search(qvec, 30), self.text_search(query, 30)):
            for rank, (cid, _) in enumerate(ranking, 1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        best = sorted(fused.items(), key=lambda x: -x[1])[:k]
        out = []
        for cid, sc in best:
            d = self.get(cid)
            d["score"] = sc
            out.append(d)
        return out
