#!/usr/bin/env python3
"""Fetch YTU Bologna syllabus pages for the exam-scope courses.

The Bologna course pages are in Turkish. This script does *faithful extraction
only*: section/field labels are mapped to English (they are a fixed enumerable
set), but free-text values (objective, content, weekly topics, outcomes) are
kept in their original Turkish. Output is staged under ``tmp/syllabi_raw/``.

The final English ``written/<course>/syllabus.md`` files are authored from this
staging by translating the Turkish prose. Translation is intentionally NOT
automated here (there is no offline TR->EN translator in the env; per the repo
convention, Claude handles translation). Extraction stays reproducible:

    python3 scripts/fetch_syllabi.py        # refresh staging from the live site

Dependencies: requests, beautifulsoup4 (see requirements.txt).
"""
from __future__ import annotations

import pathlib
import sys

import requests
import urllib3
from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tmp" / "syllabi_raw"
URL = "https://bologna.yildiz.edu.tr/index.php?r=course/view&id={cid}&aid=18&pid=266"

# (course code, Bologna course id, target folder under written/)
COURSES = [
    ("KOM5106", 7175, "KOM5106-System-Analysis-Techniques"),
    ("KOM5107", 6873, "KOM5107-System-Dynamics-Modeling-and-Simulation"),
    ("KOM5111", 9392, "KOM5111-Data-Communication-in-Automation-Systems"),
    ("KOM6115", 11635, "KOM6115-Reinforcement-Learning-Based-Optimal-Control"),
    ("KOM6203", 7172, "KOM6203-System-Theory"),
    ("KOM6110", 6854, "KOM6110-Machine-Learning-and-Artificial-Neural-Networks"),
]

HEADINGS = ("h1", "h2", "h3", "h4")

# Fixed Turkish field labels -> English (free-text values are left untranslated).
LABELS = {
    "Ders Adı": "Course Name",
    "Kodu": "Code",
    "Yerel Kredi": "Local Credit",
    "AKTS": "ECTS",
    "Ders (saat/hafta)": "Lecture (h/week)",
    "Uygulama (saat/hafta)": "Tutorial (h/week)",
    "Laboratuar (saat/hafta)": "Lab (h/week)",
    "Önkoşullar": "Prerequisites",
    "Yarıyıl": "Semester",
    "Dersin Dili": "Language of Instruction",
    "Dersin Seviyesi": "Course Level",
    "Dersin Türü": "Course Type",
    "Ders Kategorisi": "Course Category",
    "Dersin Veriliş Şekli": "Mode of Delivery",
    "Dersi Sunan Akademik Birim": "Offering Department",
    "Dersin Koordinatörü": "Course Coordinator",
    "Dersi Veren(ler)": "Instructor(s)",
    "Asistan(lar)ı": "Assistant(s)",
    "Dersin Amacı": "Course Objective",
    "Dersin İçeriği": "Course Content",
    "Ders Kitabı / Malzemesi / Önerilen Kaynaklar": "Textbook / Materials / References",
    "Opsiyonel Program Bileşenleri": "Optional Program Components",
}


def fetch(url: str) -> str:
    """GET the page. YTU serves an incomplete TLS chain (missing intermediate);
    certifi cannot verify it although the root is trusted, so fall back to an
    unverified request — exactly what the working ``curl`` does here."""
    try:
        r = requests.get(url, timeout=30)
    except requests.exceptions.SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.get(url, timeout=30, verify=False)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def find_heading(soup: BeautifulSoup, needle: str):
    return soup.find(lambda t: t.name in HEADINGS and needle in t.get_text())


def rows_of(table) -> list[list[str]]:
    out = []
    for tr in table.find_all("tr"):
        out.append([c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])])
    return out


def extract(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    lo_head = find_heading(soup, "Öğrenim Çıktıları")

    # General info: first table is a 7-col header/value pair; the following
    # 2-column rows (up to the learning-outcomes heading) are key/value fields.
    head = {}
    first = soup.find("table")
    frows = rows_of(first)
    if len(frows) >= 2:
        head = dict(zip(frows[0], frows[1]))

    info: dict[str, str] = {}
    for tbl in reversed(lo_head.find_all_previous("table")):
        for cells in rows_of(tbl):
            if len(cells) == 2 and cells[0]:
                info[cells[0]] = cells[1]

    # Learning outcomes: ordered list right after the heading.
    outcomes = []
    ol = lo_head.find_next("ol")
    if ol is not None:
        outcomes = [li.get_text(" ", strip=True) for li in ol.find_all("li")]

    def table_after(needle: str) -> list[list[str]]:
        h = find_heading(soup, needle)
        return rows_of(h.find_next("table")) if h else []

    return {
        "head": head,
        "info": info,
        "outcomes": outcomes,
        "weekly": table_after("Haftalık"),
        "assessment": table_after("Değerlendirme"),
    }


def to_markdown(code: str, cid: int, data: dict) -> str:
    L = LABELS
    head, info = data["head"], data["info"]
    md = [f"# {code} — Syllabus (raw extraction)",
          "",
          f"Source: {URL.format(cid=cid)}",
          "",
          "> Field labels translated to English; free-text values are the",
          "> original Turkish, to be translated when authoring the final file.",
          "",
          "## General information",
          ""]
    for k in ("Ders Adı", "Kodu", "Yerel Kredi", "AKTS",
              "Ders (saat/hafta)", "Uygulama (saat/hafta)", "Laboratuar (saat/hafta)"):
        if k in head:
            md.append(f"- **{L.get(k, k)}:** {head[k]}")
    for k, v in info.items():
        if k in ("Dersin Amacı", "Dersin İçeriği",
                 "Ders Kitabı / Malzemesi / Önerilen Kaynaklar"):
            continue  # rendered as their own sections below
        md.append(f"- **{L.get(k, k)}:** {v}")

    for key, title in (("Dersin Amacı", "Course Objective"),
                       ("Dersin İçeriği", "Course Content"),
                       ("Ders Kitabı / Malzemesi / Önerilen Kaynaklar",
                        "Textbook / Materials / References")):
        if info.get(key):
            md += ["", f"## {title}", "", info[key]]

    if data["outcomes"]:
        md += ["", "## Learning outcomes", ""]
        md += [f"{i}. {o}" for i, o in enumerate(data["outcomes"], 1)]

    def render_table(title: str, rows: list[list[str]]):
        if not rows:
            return
        md.extend(["", f"## {title}", ""])
        header, *body = rows
        md.append("| " + " | ".join(header) + " |")
        md.append("| " + " | ".join("---" for _ in header) + " |")
        for r in body:
            r = r + [""] * (len(header) - len(r))
            md.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")

    render_table("Weekly topics", data["weekly"])
    render_table("Assessment system", data["assessment"])
    return "\n".join(md) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for code, cid, folder in COURSES:
        html = fetch(URL.format(cid=cid))
        data = extract(html)
        out = OUT_DIR / f"{code}.md"
        out.write_text(to_markdown(code, cid, data), encoding="utf-8")
        print(f"{code}: weekly={len(data['weekly'])-1 if data['weekly'] else 0} rows, "
              f"outcomes={len(data['outcomes'])}  ->  {out.relative_to(ROOT)}")
    print(f"\nStaged {len(COURSES)} syllabi under {OUT_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
