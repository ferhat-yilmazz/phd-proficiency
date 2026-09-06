#!/usr/bin/env python3
"""Extract per-page plain text from lecture/exam PDFs.

Usage:
    python3 scripts/extract_pdf_text.py <pdf> [<pdf> ...] --out-dir tmp/extract

Writes one `<stem>.txt` per input PDF, with a `=== page N ===` marker before
each page so slide boundaries stay visible in the extracted text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def extract(pdf_path: Path, out_dir: Path) -> Path:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for number, page in enumerate(reader.pages, start=1):
        chunks.append(f"=== page {number} ===")
        chunks.append(page.extract_text() or "")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}.txt"
    out_path.write_text("\n".join(chunks), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files to extract")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("tmp/extract"), help="output directory"
    )
    args = parser.parse_args()

    for pdf in args.pdfs:
        out = extract(pdf, args.out_dir)
        print(f"{pdf} -> {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
