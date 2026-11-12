"""Build an editable .docx letter from a Markdown source.

Used for the JFDS cover letter and the declaration-of-interests page. The
internal header above the first '---' separator is skipped.

Usage: py code/make_cover_letter_docx.py [source.md] [output.docx]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "M7-JFDS-Cover-Letter.md"
OUT = ROOT / "M7-JFDS-Cover-Letter.docx"


def add_runs(paragraph, text: str) -> None:
    for chunk in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            run = paragraph.add_run(chunk[2:-2])
            run.bold = True
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            run = paragraph.add_run(chunk[1:-1])
            run.italic = True
        else:
            paragraph.add_run(chunk)


def add_rule(doc) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "999999")
    border.append(bottom)
    p._p.get_or_add_pPr().append(border)


def build(src: Path = SRC, out: Path = OUT, compact_first_block: bool = True) -> None:
    raw = src.read_text(encoding="utf-8")
    if "<!-- BODY -->" in raw:
        body = raw.split("<!-- BODY -->", 1)[1].lstrip("\n")
    else:
        body = raw.split("\n---\n", 1)[1]
    if body.startswith("---"):
        body = body.split("\n", 1)[1]

    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(8)

    # sender block is set flush right until the date line
    blocks = [b for b in re.split(r"\n\s*\n", body.strip()) if b.strip()]
    for idx, block in enumerate(blocks):
        if block.strip() == "---":
            add_rule(doc)
            continue
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        if idx == 0 and compact_first_block:
            for ln in lines:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                add_runs(p, ln.replace("`", ""))
            doc.add_paragraph()
            continue
        text = " ".join(lines).replace("`", "")
        p = doc.add_paragraph()
        add_runs(p, text)

    doc.save(out)
    print(f"WROTE {out}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    build(src, out, compact_first_block="Cover-Letter" in src.name)
