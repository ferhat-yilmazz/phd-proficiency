# KOM6110 Lecture Notes

Notes for **Machine Learning and Artificial Neural Networks**, Dr. Muharrem Mercimek, YTU.

## Week numbering

The lecturer's weekly plan (`tmp/KOM6110-syllabus.png`, reproduced on slide 8 of the Week-01 deck) and the slide **filenames** use different numbering. These notes follow the weekly plan. The mapping:

| Note | Topic | Slide deck in `resource/ytu/` |
|---|---|---|
| [Week 1](week-01-introduction-and-neuron-models.md) | Introduction: what machine learning is, ANN representations, biological motivation | `Week-01-Introduction` |
| [Week 2](week-02-knowledge-representation-and-learning.md) | Knowledge representation, learning types in brief | `Week-02-Intro-Knowledge-Learning` |
| [Week 3](week-03-bayes-decision-theory-i.md) | Bayes decision theory: Bayes classifiers (1) | `Week-05-Bayes` |
| [Week 4](week-04-bayes-classifiers-ii-discriminant-functions.md) | Bayes decision theory: Bayes classifiers (2) | `Week-06-Bayes-II` |

Remaining weeks in the plan, not yet written up: 5 (dimensionality reduction, PCA), 6 (instance based learning, kernel methods, clustering), 7 and 9 (perceptrons, MLPs, backpropagation), 10 (linear discriminant functions, SVM), 11 and 12 (reinforcement learning), 13 and 14 (CNNs).

## Sources

Primary material is the YTU slide decks under `../resource/ytu/`. The decks in turn draw on:

- Haykin, *Neural Networks and Learning Machines*, 3rd ed., 2009 (weeks 1 and 2)
- Duda, Hart and Stork, *Pattern Classification*, 2nd ed., 2000 (weeks 3 and 4)
- Alpaydin, *Introduction to Machine Learning*, 2nd ed., 2004

Slide text was extracted reproducibly with `python3 scripts/extract_pdf_text.py <pdf> --out-dir tmp/extract`.

## Reading the math

Equations use LaTeX inside `$...$` and `$$...$$`, which GitHub renders natively. Local Markdown previewers may need a MathJax or KaTeX extension.

## Conventions used throughout

- $P(\cdot)$ is a probability, $p(\cdot)$ a probability density.
- Vectors are column vectors and bold: $\mathbf{x}$, $\boldsymbol{\mu}$. Matrices are bold capitals: $\boldsymbol{\Sigma}$.
- $\omega_j$ is a state of nature (class), $\alpha_i$ an action, $\lambda(\alpha_i \mid \omega_j)$ a loss.
- The bias is treated as weight $w_{k0}$ on an input fixed at $x_0 = +1$.

Where a slide contains a transcription error, the note keeps the corrected form and records the original in a short table, so nothing is silently changed.
