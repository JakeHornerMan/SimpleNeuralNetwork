# Project Overview — SimpleNeuralNetwork

A learn-by-building project: a feed-forward neural network for **MNIST handwritten-digit classification**, assembled one step at a time in PyTorch. The code is deliberately over-commented (a line-by-line comment on nearly everything) because the goal is understanding, not just a working model.

**Architecture being built:** `784 → 128 → 10`
`Flatten` → `Linear(784, 128)` → `ReLU` → `Linear(128, 10)` → logits

---

## Progress so far

### ✅ Step 1 — Load the data · `load_mnist_data.py`
- Uses `torchvision.datasets.MNIST` to fetch the **60,000** training and **10,000** test images (auto-downloaded into `./data`, cached after the first run).
- Applies `transforms.ToTensor()` — converts each 28×28 image to a tensor and rescales pixels from 0–255 to **0.0–1.0**.
- Prints a sanity check confirming: 60000 / 10000 counts, image shape `[1, 28, 28]`, and the first label (a `5`). **Verified working.**

### ✅ Step 2 — Define the model · `model.py`
- `SimpleNN(nn.Module)` with the architecture above.
- `forward()` returns **raw logits — no softmax**, by design, because the planned loss `nn.CrossEntropyLoss` applies softmax internally.
- Confirmed it instantiates and outputs the correct `[batch, 10]` shape.

### ✅ Bonus — Interactive front-end · `nn_visualizer.html`
- A self-contained HTML/JS Artifact that visualizes the model: **draw a digit** and watch it flow through the network live.
- The 784 input pixels are abstracted into the single drawn image; the 128 hidden neurons and 10 outputs are drawn as circles that **light up with their real activations** (the forward-pass math is reimplemented in JavaScript with PyTorch-style random weight init).
- Neuron/line brightness **fades** with activation strength (strongest = **green**); the winning output digit (argmax) is highlighted in **amber**.
- Honestly labeled as **untrained** — predictions are random because no training has happened yet. Live at the Artifact URL; the file is version-controlled in the repo.

### 📄 Documentation
- `CLAUDE.md` — guidance for future Claude Code sessions (commands, architecture, conventions, what's intentionally unbuilt).

---

## Not built yet

- **Step 3 — Training loop:** `DataLoader` batching, optimizer (e.g. SGD/Adam), `CrossEntropyLoss`, the epoch loop, and saving the trained weights.
- **Step 4 — Evaluation:** measuring accuracy on the 10k test set.
- **Wiring trained weights into `nn_visualizer.html`** so it becomes a real digit classifier instead of a random one.

---

## How to run

```bash
pip install torch torchvision     # one-time setup
python load_mnist_data.py         # download data + sanity check
python -c "import torch; from model import SimpleNN; print(SimpleNN()(torch.randn(4,1,28,28)).shape)"   # check the model → [4, 10]
```

**Environment:** Python 3.11, PyTorch 2.x, Windows.
