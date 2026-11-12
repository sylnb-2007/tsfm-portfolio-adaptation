"""Build a Journal of Forecasting (Wiley) style editable .docx from the Markdown source.

This is a submission-preparation helper, not research code. The journal accepts free-format
submissions, so the file follows only the requirements that matter at first submission: a
title page with running title and affiliations, an abstract, up to six keywords, APA
references, tables with titles, and figure legends. Equations are rendered as centered
text and must be re-typeset in the Word Equation Editor before acceptance.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "M7-JOF-投稿正文.md"
OUT = ROOT / "M7-JOF-投稿正文.docx"

XSL_CANDIDATES = [
    Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files (x86)\Microsoft Office\Office16\MML2OMML.XSL"),
]


def load_xsl():
    for candidate in XSL_CANDIDATES:
        if candidate.exists():
            return etree.XSLT(etree.parse(str(candidate)))
    return None

EQUATIONS = {
    r"y_{i,t}^{(h)} = \prod_{k=1}^{h}\left(1+r_{i,t+k}\right)-1,": (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        '<msubsup><mi>y</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow>'
        '<mrow><mo>(</mo><mi>h</mi><mo>)</mo></mrow></msubsup>'
        '<mo>=</mo>'
        '<munderover><mo>&#8719;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow>'
        '<mi>h</mi></munderover>'
        '<mrow><mo>(</mo><mn>1</mn><mo>+</mo>'
        '<msub><mi>r</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi><mo>+</mo><mi>k</mi></mrow></msub>'
        '<mo>)</mo></mrow>'
        '<mo>&#8722;</mo><mn>1</mn></mrow></math>'
    ),
    r"w_{i,t} = 2\,\frac{\operatorname{rank}(p_{i,t})-\overline{\operatorname{rank}}}{\sum_j\left|\operatorname{rank}(p_{j,t})-\overline{\operatorname{rank}}\right|},": (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        '<msub><mi>w</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub>'
        '<mo>=</mo><mn>2</mn>'
        '<mfrac><mrow>'
        '<mi mathvariant="normal">rank</mi><mo>(</mo>'
        '<msub><mi>p</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>)</mo>'
        '<mo>&#8722;</mo>'
        '<mover accent="true"><mi mathvariant="normal">rank</mi><mo>&#175;</mo></mover>'
        '</mrow><mrow>'
        '<munder><mo>&#8721;</mo><mi>j</mi></munder>'
        '<mrow><mo>|</mo><mi mathvariant="normal">rank</mi><mo>(</mo>'
        '<msub><mi>p</mi><mrow><mi>j</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>)</mo>'
        '<mo>&#8722;</mo>'
        '<mover accent="true"><mi mathvariant="normal">rank</mi><mo>&#175;</mo></mover>'
        '<mo>|</mo></mrow>'
        '</mrow></mfrac></mrow></math>'
    ),
    r"\tau_t=\sum_i\left|w_{i,t}-w_{i,t-h}\right|,": (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        '<msub><mi>&#964;</mi><mi>t</mi></msub><mo>=</mo>'
        '<munder><mo>&#8721;</mo><mi>i</mi></munder>'
        '<mrow><mo>|</mo>'
        '<msub><mi>w</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub>'
        '<mo>&#8722;</mo>'
        '<msub><mi>w</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi><mo>&#8722;</mo><mi>h</mi></mrow></msub>'
        '<mo>|</mo></mrow></mrow></math>'
    ),
    r"\rho_t=\operatorname{Spearman}\left(p_{\cdot,t},y_{\cdot,t}^{(h)}\right).": (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        '<msub><mi>&#961;</mi><mi>t</mi></msub><mo>=</mo>'
        '<mi mathvariant="normal">Spearman</mi><mo>(</mo>'
        '<msub><mi>p</mi><mrow><mo>&#183;</mo><mo>,</mo><mi>t</mi></mrow></msub>'
        '<mo>,</mo>'
        '<msubsup><mi>y</mi><mrow><mo>&#183;</mo><mo>,</mo><mi>t</mi></mrow>'
        '<mrow><mo>(</mo><mi>h</mi><mo>)</mo></mrow></msubsup>'
        '<mo>)</mo></mrow></math>'
    ),
    r"\tilde{x}_{i,t}=2\,\frac{\operatorname{rank}(x_{i,t})}{N+1}-1,": (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        '<msub><mover accent="true"><mi>x</mi><mo>~</mo></mover>'
        '<mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub>'
        '<mo>=</mo>'
        '<mfrac><mrow><mn>2</mn><mi mathvariant="normal">rank</mi><mo>(</mo>'
        '<msub><mi>x</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>)</mo></mrow>'
        '<mrow><mi>N</mi><mo>+</mo><mn>1</mn></mrow></mfrac>'
        '<mo>&#8722;</mo><mn>1</mn></mrow></math>'
    ),
}

FALLBACK = {
    r"y_{i,t}^{(h)} = \prod_{k=1}^{h}\left(1+r_{i,t+k}\right)-1,":
        "y(i,t)^(h) = \u220f(k = 1 to h) (1 + r(i,t+k)) \u2212 1",
    r"w_{i,t} = 2\,\frac{\operatorname{rank}(p_{i,t})-\overline{\operatorname{rank}}}{\sum_j\left|\operatorname{rank}(p_{j,t})-\overline{\operatorname{rank}}\right|},":
        "w(i,t) = 2 \u00b7 [rank(p(i,t)) \u2212 mean rank] / \u03a3(j) |rank(p(j,t)) \u2212 mean rank|",
    r"\tau_t=\sum_i\left|w_{i,t}-w_{i,t-h}\right|,":
        "\u03c4(t) = \u03a3(i) |w(i,t) \u2212 w(i,t\u2212h)|",
    r"\rho_t=\operatorname{Spearman}\left(p_{\cdot,t},y_{\cdot,t}^{(h)}\right).":
        "\u03c1(t) = Spearman(p(\u00b7,t), y(\u00b7,t)^(h))",
    r"\tilde{x}_{i,t}=2\,\frac{\operatorname{rank}(x_{i,t})}{N+1}-1,":
        "x\u0303(i,t) = 2 \u00b7 rank(x(i,t)) / (N + 1) \u2212 1",
}

INLINE_MATH = {
    r"$r_{i,t}$": "r(i,t)",
    r"$p_{i,t}$": "p(i,t)",
    r"$c\,\tau_t$": "c\u00b7\u03c4(t)",
    r"$\rho_t^{A}-\rho_t^{B}$": "\u03c1(A,t) \u2212 \u03c1(B,t)",
}


def clean_inline(text: str) -> str:
    for k, v in INLINE_MATH.items():
        text = text.replace(k, v)
    text = text.replace("\\operatorname", "").replace("\\,", " ")
    return text


def add_runs(paragraph, text: str) -> None:
    """Add text with **bold**, *italic*, and `code` spans."""
    text = clean_inline(text)
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    for chunk in pattern.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            run = paragraph.add_run(chunk[2:-2])
            run.bold = True
        elif chunk.startswith("`") and chunk.endswith("`"):
            run = paragraph.add_run(chunk[1:-1])
            run.font.name = "Consolas"
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            run = paragraph.add_run(chunk[1:-1])
            run.italic = True
        else:
            paragraph.add_run(chunk)


def parse_table(lines: list[str]):
    rows = []
    aligns: list[str] = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            aligns = [
                "center" if c.startswith(":") and c.endswith(":")
                else "right" if c.endswith(":")
                else "left"
                for c in cells
            ]
            continue
        rows.append(cells)
    return rows, aligns


def drop_empty_paragraphs(cell) -> None:
    """Horizontal/vertical merges concatenate paragraphs; keep only non-empty ones."""
    paragraphs = cell.paragraphs
    for paragraph in paragraphs[1:]:
        if not paragraph.text.strip():
            paragraph._element.getparent().remove(paragraph._element)


def tidy_table(table, rows: list[list[str]]) -> None:
    """Merge panel-header rows horizontally and group-label cells vertically."""
    ncols = len(table.columns)
    for r_idx, row in enumerate(rows):
        if r_idx == 0:
            continue
        nonempty = [c_idx for c_idx, value in enumerate(row) if value]
        if len(nonempty) == 1 and nonempty[0] == 0 and ncols > 1:
            merged = table.cell(r_idx, 0).merge(table.cell(r_idx, ncols - 1))
            drop_empty_paragraphs(merged)
    r_idx = 1
    while r_idx < len(rows):
        if rows[r_idx][0] == "" and rows[r_idx - 1][0] != "":
            end = r_idx
            while end + 1 < len(rows) and rows[end + 1][0] == "":
                end += 1
            merged = table.cell(r_idx - 1, 0).merge(table.cell(end, 0))
            drop_empty_paragraphs(merged)
            r_idx = end + 1
        else:
            r_idx += 1


def three_line(table) -> None:
    """Apply the journal-style three-line rule: top, header rule, bottom only."""
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "8")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    for edge in ("left", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "auto")
        borders.append(el)
    tbl_pr.append(borders)
    for cell in table.rows[0].cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_borders = OxmlElement("w:tcBorders")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:space"), "0")
        bottom.set(qn("w:color"), "000000")
        tc_borders.append(bottom)
        tc_pr.append(tc_borders)


def set_table_cell_margins(table, left_cm: float = 0.08, right_cm: float = 0.08,
                           top_cm: float = 0.02, bottom_cm: float = 0.02) -> None:
    """Tighten cell padding so numeric columns do not wrap."""
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for edge, value in (("top", top_cm), ("left", left_cm), ("bottom", bottom_cm), ("right", right_cm)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(int(value * 567)))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tbl_pr.append(mar)


def set_column_widths(table, rows: list[list[str]], total_cm: float = 16.5,
                      font_pt: float = 9.0) -> None:
    """Size columns from the longest unbreakable token, then share the slack by text volume."""
    char_cm = font_pt * 0.0353 * 0.55
    pad = 0.34
    mins, wants = [], []
    for c_idx in range(len(table.columns)):
        cells = [r[c_idx] for r in rows if c_idx < len(r)]
        tokens = [tok for cell in cells for tok in cell.split()]
        longest = max((len(tok) for tok in tokens), default=1)
        avg = sum(len(cell) for cell in cells) / max(len(cells), 1)
        mins.append(longest * char_cm + pad)
        wants.append(min(avg, longest * 1.6) * char_cm + pad)
    total_min = sum(mins)
    if total_min >= total_cm:
        widths = [m * total_cm / total_min for m in mins]
    else:
        extra = total_cm - total_min
        want_sum = sum(wants)
        widths = [mins[idx] + extra * wants[idx] / want_sum for idx in range(len(mins))]
    for c_idx, width in enumerate(widths):
        c_width = Cm(width)
        table.columns[c_idx].width = c_width
        for cell in table.columns[c_idx].cells:
            cell.width = c_width


def add_page_number(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, instr, end):
        run._r.append(node)


def add_black_heading(doc, text: str, size: int = 14) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(12)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)


def build() -> None:
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    add_page_number(doc.sections[0])

    lines = SRC.read_text(encoding="utf-8").splitlines()
    xsl = load_xsl()
    if xsl is None:
        print("WARNING: MML2OMML.XSL not found; equations will be plain text", file=sys.stderr)
    i = 0
    started_body = False
    in_references = False
    while i < len(lines):
        line = lines[i].rstrip()

        if line.strip() == "## Figure and table captions":
            break

        if not line.strip() or line.strip() == "---":
            i += 1
            continue

        # equation block
        if line.strip() == "$$":
            j = i + 1
            chunk = []
            while j < len(lines) and lines[j].strip() != "$$":
                chunk.append(lines[j].strip())
                j += 1
            body = " ".join(chunk)
            p = None
            if body in EQUATIONS and xsl is not None:
                try:
                    result = xsl(etree.fromstring(EQUATIONS[body].encode()))
                    root = result.getroot()
                    if root.tag.endswith("}oMathPara") and len(root):
                        root = root[0]
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p._p.append(parse_xml(etree.tostring(root, encoding="unicode")))
                except Exception as exc:  # noqa: BLE001
                    print(f"OMML fallback for equation: {exc}", file=sys.stderr)
                    p = None
            if p is None:
                p = doc.add_paragraph(FALLBACK.get(body, body))
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.name = "Cambria Math"
            i = j + 1
            continue

        # headings
        if line.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(18)
            run = p.add_run(line[2:].strip())
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(16)
            started_body = True
            i += 1
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            in_references = title.lower() == "references"
            p = doc.add_heading(title, level=1)
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            for run in p.runs:
                run.bold = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(13)
                run.font.color.rgb = RGBColor(0, 0, 0)
            started_body = True
            i += 1
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            p = doc.add_heading(title, level=2)
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            for run in p.runs:
                run.bold = True
                run.italic = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(12)
                run.font.color.rgb = RGBColor(0, 0, 0)
            i += 1
            continue

        # table
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows, aligns = parse_table(block)
            if rows:
                ncols = max(len(r) for r in rows)
                tbl_font = 8.5 if ncols >= 7 else 9.0
                table = doc.add_table(rows=len(rows), cols=ncols)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.autofit = False
                for r_idx, row in enumerate(rows):
                    for c_idx in range(ncols):
                        cell = table.cell(r_idx, c_idx)
                        cell.text = ""
                        p = cell.paragraphs[0]
                        p.paragraph_format.line_spacing = 1.0
                        p.paragraph_format.space_after = Pt(0)
                        align = aligns[c_idx] if c_idx < len(aligns) else "left"
                        p.alignment = {
                            "right": WD_ALIGN_PARAGRAPH.RIGHT,
                            "center": WD_ALIGN_PARAGRAPH.CENTER,
                        }.get(align, WD_ALIGN_PARAGRAPH.LEFT)
                        value = row[c_idx] if c_idx < len(row) else ""
                        add_runs(p, value)
                        for run in p.runs:
                            run.font.size = Pt(tbl_font)
                            if r_idx == 0:
                                run.bold = True
                tidy_table(table, rows)
                three_line(table)
                set_table_cell_margins(table)
                set_column_widths(table, rows, font_pt=tbl_font)
                for row in table.rows:
                    tr_pr = row._tr.get_or_add_trPr()
                    tr_pr.append(OxmlElement("w:cantSplit"))
                spacer = doc.add_paragraph()
                spacer.paragraph_format.space_before = Pt(0)
                spacer.paragraph_format.space_after = Pt(4)
                spacer.paragraph_format.line_spacing = 1.0
            continue

        # bullet list
        if line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, line[2:].strip())
            i += 1
            continue

        # paragraph
        p = doc.add_paragraph()
        if in_references:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, line.strip())
        if line.strip().startswith(("*Notes:*", "Notes:")):
            p.paragraph_format.space_before = Pt(2)
            for run in p.runs:
                run.font.size = Pt(9)
        if line.strip().startswith("**Table"):
            p.paragraph_format.keep_with_next = True
        i += 1

    # Figures section
    doc.add_page_break()
    add_black_heading(doc, "Figures", size=14)
    caption_lines = SRC.read_text(encoding="utf-8").splitlines()
    for cap in caption_lines:
        if cap.startswith("**Figure"):
            text = re.sub(r"\s+\*(Explanatory|Result) figure;[^$]*?\*\s*", " ", cap)
            text = re.sub(r"\s*Files:\s*.*$", "", text).strip()
            p = doc.add_paragraph()
            add_runs(p, text)
            for run in p.runs:
                run.font.size = Pt(10)
            match = re.search(r"`?(figures/(fig[0-9]+_[a-z_]+)\.png)`?", cap)
            if match:
                png = ROOT / "figures" / f"{match.group(2)}.png"
                if png.exists():
                    doc.add_picture(str(png), width=Inches(6.0))
                    doc.add_paragraph()

    doc.save(OUT)
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
