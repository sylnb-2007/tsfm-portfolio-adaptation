"""Cross-reference audit for the JFDS submission manuscript.

Checks heading order, figure/table caption uniqueness, in-text references, and
placeholder / word counts. Prints a plain-text report.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "M7-JFDS-投稿正文.md"


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    print("--- headings ---")
    for line in text.splitlines():
        if re.match(r"^#{1,3} ", line):
            print(line.strip())

    print("\n--- captions vs mentions ---")
    for kind, count in (("Table", 10), ("Figure", 5)):
        for n in range(1, count + 1):
            caption = len(re.findall(rf"\*\*{kind} {n}\.", text))
            mention = len(re.findall(rf"(?<![\*A-Za-z]){kind} {n}(?!\.)(?![0-9])", text))
            flag = "" if caption == 1 else "  <-- CHECK CAPTION"
            print(f"{kind} {n}: captions={caption} mentions={mention}{flag}")

    print("\n--- style checks ---")
    print("Fig. abbreviation:", len(re.findall(r"Fig\.\s", text)))
    print("words:", len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2019-]*", text)))
    print("placeholders:", text.count("AUTHOR_INPUT_NEEDED"))


if __name__ == "__main__":
    main()
