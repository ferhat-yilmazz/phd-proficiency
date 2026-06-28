# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A personal study workspace for preparing for a **PhD proficiency / qualifying exam**
for the **Control & Automation Engineering** at the Yildiz Technical University.
It is not a software product. Claude is used to generate course notes and exercises mostly.

The usual "build / lint / test" lifecycle therefore does not apply. The only "code"
is self-contained Python helpers (see *Python environment*) plus written study material.

### Exam scope

`course_list.txt` lists the full KOM/BLM graduate catalog and the subset selected as
the exam scope. The selected courses:

| Code | Course (EN) | Area |
|------|-------------|------|
| KOM5106 | System Analysis Techniques | dynamic-system modeling & analysis |
| KOM5111 | Data Communication in Automation Systems | industrial networks / fieldbus |
| KOM6115 | Reinforcement-Learning-Based Optimal Control | optimal control, dynamic programming, RL |
| KOM6203 | System Theory | linear systems, state-space, controllability/observability, stability |
| KOM6110 | Machine Learning and Artificial Neural Networks | ML, neural networks |

Past papers in `prev_data/` use the tags **ST** = System Theory and **SAT** = System
Analysis Techniques.

## Working language

Produce all output — notes, summaries, code comments, filenames, docs, commit
messages — in **English (US)**. If any material presented as **Turkish**;
translate it to English when summarizing or quoting, unless explicitly asked to
preserve the original Turkish.

## Python environment

All helper scripts and tooling are **Python**, and must run inside the dedicated
**pyenv-virtualenv `phd-proficiency`** (Python 3.14.6). It auto-activates from
`.python-version` whenever the shell's working directory is the project root, so
`python3` / `pip` already resolve to the correct interpreter — no manual activation needed.

```bash
# sanity check — should print .../versions/phd-proficiency
python3 -c "import sys; print(sys.prefix)"

# install / update dependencies
python3 -m pip install -r requirements.txt

# run a helper (from the project root, so the env is active)
python3 scripts/<name>.py
```

Rules:
- Never install into or run against the system Python; always the `phd-proficiency` env.
- Record every new third-party dependency in `requirements.txt`.
- If `python3` ever points outside the env, run `pyenv activate phd-proficiency` from the project root.

## Layout

- `.claude/` — all Claude Code configuration for this project (`settings.json`; project
  skills/commands also belong here).
- `written/` — study material, notes, and practice for the **written** exam. Internal
  structure (by course/topic) is decided as material is added.
- `oral/` — study material, notes, and practice for the **oral** exam. Internal
  structure is decided as material is added.
- `scripts/` — Python helper scripts and tools (extract text/figures from exam files,
  generate practice problems, check derivations, etc.).
- `tmp/` — scratch space for all temporary/intermediate work; git-ignored (not committed).
  Shared: both Claude and the engineer may write here.
- `prev_data/` — **read-only reference material** (writes are blocked in
  `.claude/settings.json`). Past exams, lecture PDFs, and annotated slides — treat as
  ground-truth inputs; never modify or delete:
  - `28_kasım_2022_yeterlilik_soru_ST_SAT.docx` — 28 Nov 2022 exam (System Theory + System Analysis Techniques)
  - `lineer_sistemler.pdf` — Linear Systems lecture notes (~22 MB)
  - `written_exam.jpeg`, `oral_exam.docx` — written & oral exam material
  - `l01-ann/` — Lecture 1 (ANN) annotated slide screenshots
- `course_list.txt` — full course catalog + selected exam-scope courses.
- `CLAUDE.md` — this file.

## Conventions

- Parse `.docx` / `.pdf` / image sources with a committed Python script (e.g.
  `python-docx`, `pypdf`) rather than one-off shell, so extraction is reproducible.
- Keep generated study artifacts out of `prev_data/`; write study notes under
  `written/` or `oral/`, and script outputs under `scripts/` paths.
- The exam has two stages — **written** and **oral** — mirrored by the `written/` and
  `oral/` top-level folders. Place material in whichever stage it supports.

  ## Workflow
  - The engineer presents course list and syllabus
  - The engineer presents available course materials
  - The engineer request detailed course notes; Claude shall research on web for additional note if required
  - The engineer request exercise questions; Claude shall generate questions according to course material and notes

  ## Project Rules
  - Claude shall detect knowledge level of the engineer by asking questions about a course
  - Claude shall generate notes and questions as Markdown files
  - Since the engineer will use Github to render Markdown files, all equations and mathematical expressions shall be
  implemented convenient with Github
  - Claude shall use verified websites to get course notes, e.g. universities
  - Claude shall explain all topics and questions by teaching core
  - Claude shall avoid verbosity, it shall be think more to explain deeply with few words
  - Claude shall use the repository's `tmp/` folder for all temporary work, never the
  system `/tmp`. The engineer may also use `tmp/` for scratch files.
