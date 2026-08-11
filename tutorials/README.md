# Technical Tutorials

This folder is a self-contained course for understanding the ideas behind
CoDy-JEPA. It teaches the mathematics, machine learning, statistics, and
reproducible-engineering habits that the active hierarchical-diversity study
depends on.

Every lesson has two parts:

- a lecture in `tutorials/lectures/`;
- an executable notebook in `tutorials/implementations/`.

The notebooks use synthetic data, run on CPU, and do not require private
datasets or repository outputs.

![The six tutorial tracks, each supplying what the next one assumes](images/curriculum_map.svg)

## Curriculum Shape

The lessons build in order:

1. tensor shapes, vector geometry, and grouped observations;
2. JEPA representation learning and sampling discipline;
3. representation geometry and GFC-style retrieval scoring;
4. paired inference, fixed exposure, and model-level replication;
5. reproducible evaluator contracts;
6. the full hierarchical-diversity study.

Use the numbered order unless you already know the prerequisites named at the
top of a lecture.

## Folder Contents

- `lectures/` contains the conceptual explanations.
- `implementations/` contains notebooks that turn each idea into code.
- `images/` contains the diagrams used by the lectures and notebooks.
- `check_tutorials.py` validates the curriculum structure, links, figures, and
  notebook metadata.

## Setup

Run from the repository root:

```bash
uv sync --locked --group dev
uv run --locked jupyter lab tutorials/implementations
```

Validate the tutorial files with:

```bash
uv run --locked python tutorials/check_tutorials.py
```

Execute the notebooks into a temporary output directory when you want a full
runtime check:

```bash
mkdir -p /tmp/cody-jepa-tutorial-runs
uv run --locked jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=120 \
  --output-dir /tmp/cody-jepa-tutorial-runs \
  tutorials/implementations/*.ipynb
```

## How To Study

Read the lecture first, then run the matching notebook. Before each notebook
cell, predict the shape and meaning of the main arrays. When an assertion fails
after you change something, read the assertion as a statement of the concept you
just broke.

After finishing the tutorials, read the active research documentation in
[docs/](../docs/README.md).
