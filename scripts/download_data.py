"""Download the Kenneth R. French 10x10 daily portfolio archives used in the paper.

The raw files are not redistributed with this repository; this script fetches them
from the public source, verifies the archive checksums and unpacks the CSV files
into ``data/``, which is the location expected by the analysis scripts.

Usage:
    python scripts/download_data.py [--out data]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

ARCHIVES = [
    (
        "100_Portfolios_10x10_Daily_CSV.zip",
        "100_Portfolios_10x10_Daily.csv",
        "8fad24e04196408125ac739f7a065c4c799d7094f53cc48f9ee34df5405a0775",
    ),
    (
        "100_Portfolios_ME_OP_10x10_Daily_CSV.zip",
        "100_Portfolios_ME_OP_10x10_Daily.csv",
        "af2c9502df287d8f7e836c88c101d11ac2c6105ab65e7ec6b3ca98462c164af6",
    ),
    (
        "100_Portfolios_ME_INV_10x10_Daily_CSV.zip",
        "100_Portfolios_ME_INV_10x10_Daily.csv",
        "85f888114f6bff845e47e70aa22521518d36d9f391ccaffc3dfd56965faf67bf",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for archive, csv_name, expected in ARCHIVES:
        target = out / archive
        if not target.exists():
            print(f"downloading {archive} ...")
            urllib.request.urlretrieve(BASE + archive, target)
        actual = sha256(target)
        if actual != expected:
            print(
                f"WARNING: checksum mismatch for {archive}\n"
                f"  expected {expected}\n  actual   {actual}\n"
                "  The source file may have been revised; check the French library "
                "documentation before using the data.",
                file=sys.stderr,
            )
        else:
            print(f"  checksum OK: {archive}")
        with zipfile.ZipFile(target) as handle:
            handle.extractall(out)
        if not (out / csv_name).exists():
            print(f"WARNING: expected {csv_name} after unpacking {archive}", file=sys.stderr)

    print("\nData ready in", out.resolve())
    print(
        "Citation: Kenneth R. French Data Library, "
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
