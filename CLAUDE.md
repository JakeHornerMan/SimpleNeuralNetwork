# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Update OverviewPrompt.md on after every prompt change.

## What this is

A learning project building a feed-forward neural network for MNIST digit classification from scratch, step by step. It is intentionally incomplete — the pieces exist as standalone scripts, not yet wired into a pipeline. Code here favors teaching clarity over terseness: **every line carries an explanatory comment by design**. Preserve that style when editing Python files; it is the point, not clutter.

## Commands

```bash
pip install torch torchvision      # one-time setup (also installs numpy, pillow)
python load_mnist_data.py          # downloads MNIST into ./data (cached) and prints a sanity check
python train.py                    # trains 5 epochs on 60k images (~98%), saves model_weights.pth
python train_one.py <file.json>    # fine-tune on one exported drawing (one gentle SGD step, LR=0.01); default = newest instance*.json
python embed_weights.py            # embed model_weights.pth into nn_visualizer.html (base64 window.TRAINED), then redeploy
python compare_neurons.py [idx]    # contrast one hidden neuron (default 2) random-vs-trained -> neuron_<idx>_compare.png
```

`loss_function.py` (Step 3a) and `optimizer.py` (Step 3b) are standalone teaching demos. `compare_neurons.py` needs `matplotlib` (`pip install matplotlib`) plus `model_weights.pth` and the MNIST test set.

There is no test suite, linter, or build step. "Verifying" a change means running the relevant script and reading its printed output. For the visualizer's JS, extract the main `<script>` and `node --check` it; confirm embedded weights match PyTorch by replaying the JS forward pass (`h=ReLU(W1@x+b1)`, `W2@h+b2`) in Python. `model.py` has no `__main__`; check it with a one-liner, e.g.:

```bash
python -c "import torch; from model import SimpleNN; print(SimpleNN()(torch.randn(4,1,28,28)).shape)"  # expect torch.Size([4, 10])
```

## Architecture

The project is progressing through numbered steps toward a trained classifier. Current state:

- **`load_mnist_data.py`** — Step 1. Fetches the MNIST train (60k) / test (10k) splits via `torchvision.datasets.MNIST` with `ToTensor()` (scales pixels to 0–1). Standalone; run once to populate `./data`.
- **`model.py`** — Step 2. Defines `SimpleNN(nn.Module)`: `Flatten` → `Linear(784, 128)` → `ReLU` → `Linear(128, 10)`. `forward` returns **raw logits, not probabilities** — softmax is deliberately omitted because the intended loss, `nn.CrossEntropyLoss`, applies it internally. Do not add a softmax to `forward`.
- **`train.py`** — Step 3c. Trains the model (`DataLoader` batch 64, `CrossEntropyLoss`, `Adam` lr 0.001, 5 epochs) and saves `model.state_dict()` to `model_weights.pth` (~98% train acc). `train_one.py` + `embed_weights.py` are the single-instance fine-tune path (see below).
- **`nn_visualizer.html`** — a self-contained interactive front-end (published as an Artifact) that reimplements the `model.py` forward pass in JavaScript. It runs the drawing through **MNIST-style preprocessing** (crop → 20px box → centre-of-mass) before the forward pass, and can load the **real trained weights** (base64-embedded via `embed_weights.py`). **If you change `model.py`'s architecture, you must update the JS forward pass, the diagram layout, and re-run `embed_weights.py`.** The embedded weights are ~530 KB of base64 in a `<script id="trained-data">` block. It also has a **Neuron compare** card (shown when a hidden neuron is selected): the same neuron's receptive field random-vs-trained on one shared colour scale, with in-browser mean/std/smoothness/bias — the browser mirror of `compare_neurons.py` (it reuses the embedded `window.TRAINED` and a fixed-seed random snapshot; no test set, so no per-digit bars).
- **`compare_neurons.py`** — teaching script (matplotlib): contrasts one hidden neuron (default 2, optional CLI index) between a fresh random model and `model_weights.pth`. Plots both 28×28 receptive fields on a shared red→green scale, prints/annotates a metrics table (weight mean/std, **spatial smoothness** = avg 4-neighbour correlation, bias), and runs the 10k test set to bar-chart the neuron's mean post-ReLU activation per digit class. Saves `neuron_<idx>_compare.png`. The point: random init has no spatial structure — a neuron only becomes a feature detector once training correlates its weights (smoothness ~0 → clearly positive).

**Single-instance fine-tune path:** the app cannot write files, so the "Export for training" button downloads `instance-<label>.json` (normalized pixels + label); `train_one.py` applies one gentle SGD step and saves `model_weights.pth`; `embed_weights.py` pushes it back into the HTML. **Guardrail:** `train_one.py`'s `LR` is dangerous — LR 0.30 in one step collapsed test accuracy to ~10% (catastrophic forgetting); keep it ~0.01.

**Only remaining step:** Step 4 — evaluate `model_weights.pth` on the held-out 10k test set (train accuracy ≠ generalization).

## Conventions

- Python is authored against Python 3.11 and PyTorch 2.x on Windows.
- MNIST auto-downloads on first run; `./data/` and `__pycache__/` are generated artifacts, not source.
