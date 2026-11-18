"""HWPX 파일에서 본문 문단 글을 순서대로 꺼낸다 (표준 라이브러리만 사용).

HWPX는 zip 파일이고, 본문은 Contents/section0.xml, section1.xml ... 에 들어 있다.
- 머리말·꼬리말·각주 안의 글은 본문이 아니므로 뺀다.
- 표는 표 자리에 «⟨표: 칸 | 칸 ...⟩» 한 덩어리 글로 넣는다 (규정의 표는 대개 별표).
- 그림은 글이 없으므로 «⟨그림⟩» 자리표만 남긴다.
"""
from __future__ import annotations

import html
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS_P = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"
SKIP = {"header", "footer", "footNote", "endNote"}


def read_paragraphs(path: str | Path) -> list[str]:
    """HWPX 파일 → 문단 글 목록."""
    with zipfile.ZipFile(path) as z:
        sections = sorted(
            (n for n in z.namelist() if re.fullmatch(r"Contents/section\d+\.xml", n)),
            key=lambda n: int(re.search(r"\d+", n.rsplit("/", 1)[1]).group()),
        )
        paras: list[str] = []
        for sec in sections:
            root = ET.fromstring(z.read(sec))
            for p in root.findall(NS_P + "p"):  # 맨 위 문단만. 표 안 문단은 _text가 훑는다.
                paras.append(html.unescape(_text(p)))
    return paras


def _text(p: ET.Element) -> str:
    parts: list[str] = []

    def walk(el: ET.Element) -> None:
        tag = el.tag.replace(NS_P, "")
        if tag in SKIP:
            return
        if tag == "t":
            parts.append(el.text or "")
            for ch in el:
                walk(ch)
                if ch.tail:
                    parts.append(ch.tail)
            return
        if tag == "lineBreak":
            parts.append("\n")
            return
        if tag == "tab":
            parts.append(" ")
            return
        if tag == "pic":
            parts.append("⟨그림⟩")
            return
        if tag == "tbl":
            cells = []
            for tc in el.iter(NS_P + "tc"):
                cell = " ".join(_text(q) for q in tc.iter(NS_P + "p"))
                cells.append(re.sub(r"\s+", " ", cell).strip())
            parts.append("⟨표: " + " | ".join(cells) + "⟩")
            return
        for ch in el:
            walk(ch)

    for ch in p:
        walk(ch)
    return "".join(parts)
