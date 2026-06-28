# PhD Proficiency Exam — Study Workspace

Personal study workspace for the **PhD proficiency / qualifying exam** in
**Control & Automation Engineering** at **Yildiz Technical University (YTU)**.

This is not a software product. It holds course notes, derivations, and practice
problems, plus small Python helpers that extract and process source material.
Notes and exercises are authored with [Claude Code](https://claude.com/claude-code);
see [`CLAUDE.md`](CLAUDE.md) for the authoring guidance.

## Exam scope

The exam has two stages — **written** and **oral**. Selected courses
(full catalog in [`course_list.txt`](course_list.txt)):

| Code | Course | Area |
|------|--------|------|
| KOM5106 | System Analysis Techniques | Dynamic-system modeling & analysis |
| KOM5111 | Data Communication in Automation Systems | Industrial networks / fieldbus |
| KOM6115 | Reinforcement-Learning-Based Optimal Control | Optimal control, dynamic programming, RL |
| KOM6203 | System Theory | Linear systems, state-space, controllability/observability, stability |
| KOM6110 | Machine Learning and Artificial Neural Networks | ML, neural networks |

Past papers are tagged **ST** (System Theory) and **SAT** (System Analysis Techniques).

## Layout

| Path | Contents |
|------|----------|
| `written/` | Notes & practice for the **written** exam, one folder per course |
| `oral/` | Notes & practice for the **oral** exam |
| `scripts/` | Python helpers (parse `.docx`/`.pdf`/images, generate & check problems) |
| `course_list.txt` | Full YTU KOM/BLM catalog + selected exam scope |
| `CLAUDE.md` | Authoring guidance and conventions |

## Python environment

Helpers run in a dedicated **pyenv-virtualenv** `phd-proficiency` (Python 3.14.6),
which auto-activates from `.python-version` at the project root.

```bash
# sanity check — should print .../versions/phd-proficiency
python3 -c "import sys; print(sys.prefix)"

# install dependencies
python3 -m pip install -r requirements.txt

# run a helper
python3 scripts/<name>.py
```

Record every new dependency in [`requirements.txt`](requirements.txt). Never use the
system Python.

## Conventions

- All material is in **English (US)**; Turkish sources are translated when summarized.
- Notes and exercises are **GitHub-flavored Markdown**, with equations written to
  render on GitHub.
- Source files (`.docx`/`.pdf`/images) are parsed with committed Python scripts, so
  extraction stays reproducible.
