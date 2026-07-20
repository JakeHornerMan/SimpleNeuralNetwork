# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Update OverviewPrompt.md on after every prompt change.

## What this is

A learning project building a feed-forward neural network for MNIST digit classification from scratch, step by step. It is intentionally incomplete — the pieces exist as standalone scripts, not yet wired into a pipeline. Code here favors teaching clarity over terseness: **every line carries an explanatory comment by design**. Preserve that style when editing Python files; it is the point, not clutter.

## Commands

```bash
pip install torch torchvision      # one-time setup (also installs numpy, pillow)
python load_mnist_data.py          # downloads MNIST into ./data (cached after first run) and prints a sanity check
```

There is no test suite, linter, or build step. "Verifying" a change means running the relevant script and reading its printed output. `model.py` has no `__main__`; check it with a one-liner, e.g.:

```bash
python -c "import torch; from model import SimpleNN; print(SimpleNN()(torch.randn(4,1,28,28)).shape)"  # expect torch.Size([4, 10])
```

## Architecture

The project is progressing through numbered steps toward a trained classifier. Current state:

- **`load_mnist_data.py`** — Step 1. Fetches the MNIST train (60k) / test (10k) splits via `torchvision.datasets.MNIST` with `ToTensor()` (scales pixels to 0–1). Standalone; run once to populate `./data`.
- **`model.py`** — Step 2. Defines `SimpleNN(nn.Module)`: `Flatten` → `Linear(784, 128)` → `ReLU` → `Linear(128, 10)`. `forward` returns **raw logits, not probabilities** — softmax is deliberately omitted because the intended loss, `nn.CrossEntropyLoss`, applies it internally. Do not add a softmax to `forward`.
- **`nn_visualizer.html`** — a self-contained interactive front-end (published as an Artifact). It reimplements the `model.py` forward pass in JavaScript with random PyTorch-style weight init (`uniform(±1/√fan_in)`) so it faithfully mirrors the **untrained** network: signal flows, but predictions are random by design. If you change the model's architecture in `model.py`, the JS forward pass and diagram layout here must be updated to match.

**Not yet built:** the training loop (Step 3) — no optimizer, DataLoader batching, loss computation, or saved weights exist yet. There is currently no trained checkpoint, so nothing in the repo produces accurate predictions.

## Conventions

- Python is authored against Python 3.11 and PyTorch 2.x on Windows.
- MNIST auto-downloads on first run; `./data/` and `__pycache__/` are generated artifacts, not source.
