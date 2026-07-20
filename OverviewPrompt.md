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

### ✅ Step 3a — Loss function · `loss_function.py`
- The "how wrong was it?" step. Uses `nn.CrossEntropyLoss` — takes the 10 raw logits + the true label and returns **one number**: small when confident-and-right, large when confident-and-wrong.
- Scores one real MNIST digit through the untrained net (lands near **ln(10) ≈ 2.30**, the random-guess baseline) and then demos the intuition with hand-made logits (confident-right ≈ 0.001, unsure ≈ 2.30, confident-wrong ≈ 9.0). **Verified working.**
- Reinforces why `model.py` returns raw logits: `CrossEntropyLoss` applies softmax internally.

### ✅ Interactive front-end · `nn_visualizer.html`
- A self-contained HTML/JS Artifact that visualizes the model: **draw a digit** and watch it flow through the network live (forward pass reimplemented in JS with PyTorch-style random weight init).
- The 784 input pixels are abstracted into the single drawn image; the 128 hidden neurons and 10 outputs are circles that **light up with their real activations**.
- Activation strength is colour-coded on a **red → green ramp**; the strongest signals (**>90%**) get a **white outline**; the winning output digit (argmax) is highlighted in **amber**.
- **"Actual digit" picker** sets a true label and drives a live **cross-entropy loss** readout (`loss = −ln(prob of true digit)`) — the browser mirror of `loss_function.py`, with a meter and an ln(10) baseline tick.
- **Hover tooltips** give plain-language reminders on the loss label, value, meter, and hint.
- **Weight inspector:** click any neuron to see its weights as a red (negative) → green (positive) heatmap — a hidden neuron shows its 784 weights as a 28×28 receptive field; an output digit shows its 128 weights on the same 8×16 grid as the diagram. Bias is shown too.
- **Hover weight preview:** hovering any neuron pops a tooltip with a mini version of that same weight heatmap plus its bias and current activation — a quick scan without opening the full inspector.
- **Dud handling:** an all-black input is treated as no signal — activations are forced to zero (all nodes red), no prediction — rather than reading the bias-driven values.
- Honestly labeled as **untrained** — predictions are random because no training has happened yet. Live at the Artifact URL; the file is version-controlled in the repo.

### 📄 Documentation
- `CLAUDE.md` — guidance for future Claude Code sessions (commands, architecture, conventions, what's intentionally unbuilt).

---

## Not built yet

- **Step 3b — Optimizer + backprop:** use the loss to actually adjust the weights (`loss.backward()`, an optimizer like SGD/Adam, `optimizer.step()`).
- **Step 3c — Training loop:** `DataLoader` batching over epochs, and saving the trained weights.
- **Step 4 — Evaluation:** measuring accuracy on the 10k test set.
- **Wiring trained weights into `nn_visualizer.html`** so it becomes a real digit classifier instead of a random one.

---

## How to run

```bash
pip install torch torchvision     # one-time setup
python load_mnist_data.py         # download data + sanity check
python loss_function.py           # Step 3a: see cross-entropy loss react to good vs bad guesses
python -c "import torch; from model import SimpleNN; print(SimpleNN()(torch.randn(4,1,28,28)).shape)"   # check the model → [4, 10]
```

**Environment:** Python 3.11, PyTorch 2.x, Windows.
