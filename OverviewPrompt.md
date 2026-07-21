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
- Scores one real MNIST digit through the untrained net (lands near **ln(10) ≈ 2.30**, the random-guess baseline) and then demos the intuition with hand-made logits. Each loss is printed **alongside its probability** `p = exp(-loss)` (right ≈ 0.999, unsure = 0.10, wrong ≈ 0.0001), making the ln(10) baseline click: 2.30 = a 1-in-10 guess. **Verified working.**
- Reinforces why `model.py` returns raw logits: `CrossEntropyLoss` applies softmax internally.

### ✅ Step 3b — Optimiser + backprop · `optimizer.py`
- The "nudge the weights" step. Uses `torch.optim.Adam(model.parameters(), lr=0.001)` — Adam = adaptive gradient descent with momentum; `lr` is the step size (landscape/walk-downhill metaphor documented in-file).
- Demonstrates the core training ritual on **one fixed batch of 8 digits**: `optimizer.zero_grad()` → forward → `loss` → `loss.backward()` (backprop = the slope) → `optimizer.step()` (the nudge). **Verified working.**
- Log makes the descent legible: per-step drop `(▼…)` shrinks as the slope flattens, `p(true)` climbs ~0.10 → ~0.79, and it flags that "8/8 correct" arrives (step 3) long before confidence does — **correct ≠ confident**.

### ✅ Step 3c — Training loop + saved weights · `train.py`
- The real thing: `DataLoader(batch_size=64, shuffle=True)` over all 60k images, `CrossEntropyLoss` + `Adam(lr=0.001)`, **5 epochs**, standard five steps per batch. Prints avg loss + train accuracy per epoch.
- Run result: train accuracy **90.56% → 97.99%**, avg loss **0.34 → 0.067**. Saves `model_weights.pth` (`model.state_dict()`). **Verified working.**

### ✅ Interactive front-end · `nn_visualizer.html`
- A self-contained HTML/JS Artifact that visualizes the model: **draw a digit** and watch it flow through the network live (forward pass reimplemented in JS with PyTorch-style random weight init).
- The 784 input pixels are abstracted into the single drawn image; the 128 hidden neurons and 10 outputs are circles that **light up with their real activations**.
- Activation strength is colour-coded on a **red → green ramp**; the strongest signals (**>90%**) get a **white outline**; the winning output digit (argmax) is highlighted in **amber**.
- **"Actual digit" picker** sets a true label and drives a live **cross-entropy loss** readout (`loss = −ln(prob of true digit)`) — the browser mirror of `loss_function.py`, with a meter and an ln(10) baseline tick.
- **Hover tooltips** give plain-language reminders on the loss label, value, meter, and hint.
- **Weight inspector:** click any neuron to see its weights as a red (negative) → green (positive) heatmap — a hidden neuron shows its 784 weights as a 28×28 receptive field; an output digit shows its 128 weights on the same 8×16 grid as the diagram. Bias is shown too. Panel is hidden until a neuron is selected; click the same neuron again to deselect. The weight image has its own hover tooltip explaining what it represents.
- **Output colours:** the winning digit (argmax) is green with a white ring; all losing digits are red.
- **Hover weight preview:** hovering any neuron pops a tooltip with a mini version of that same weight heatmap plus its bias and current activation — a quick scan without opening the full inspector.
- **Dud handling:** an all-black input is treated as no signal — activations are forced to zero (all nodes red), no prediction — rather than reading the bias-driven values.
- **Load trained weights:** a button pours the real weights from `model_weights.pth` (base64-embedded in the page, ~530 KB) into the JS forward pass — no retraining, just the precomputed numbers. Verified: the browser math matches PyTorch on 200/200 test images (~99%). A tooltip on the button describes the training data; toggling back gives random weights. Messaging switches between "untrained" and "trained".
- **"How this network learned" card:** a section (styled like the inspector) with the training facts — 60k images, 5 epochs, batch 64, Adam lr 0.001 — plus a per-epoch accuracy bar chart (90.6% → 98.0%).
- **MNIST-style input normalization:** the drawing is cropped to the ink, scaled to a 20px box, and centred by centre-of-mass before the forward pass — otherwise trained-mode predictions are garbage on freehand strokes (verified: a raw off-centre bar predicts `6`, the same bar preprocessed predicts `1` @ 99.96%). A "what the network sees" preview shows the normalized 28×28 the net actually reads.
- Honestly labeled as **untrained** — predictions are random because no training has happened yet. Live at the Artifact URL; the file is version-controlled in the repo.

### ✅ Singular training path — fine-tune on one drawing · `train_one.py` + `embed_weights.py` + app export
- The app can't write `model_weights.pth` (browser sandbox), so single-instance correction runs through Python.
- **App:** an **"Export for training"** button (under the Actual-digit picker, enabled once a digit is drawn AND a true label is picked) downloads `instance-<label>.json` = the normalized `netInput` (784 floats) + label. Client-side Blob download, CSP-safe.
- **`train_one.py`:** loads `model_weights.pth`, reads the instance, applies **one gentle SGD step** (`LR = 0.01`), prints prediction/`p(true)`/loss before→after, saves the weights back. Reset = delete `model_weights.pth` + re-run `train.py`.
- **Critical finding (baked into comments):** LR is dangerous — one step at LR 0.30 **collapsed** test accuracy 97.5% → ~10% (catastrophic forgetting); LR 0.01 corrects the example (verified: a misclassified `6`→`4`, p 0.019→0.980) with only ~0.3% accuracy cost. Nudges are additive.
- **`embed_weights.py`:** re-embeds `model_weights.pth` as the base64 `window.TRAINED` block in `nn_visualizer.html` (idempotent). Loop: export → `train_one.py` → `embed_weights.py` → redeploy. **All verified working.**
- **"One SGD step, explained" popup:** clicking **Export for training** opens a modal that computes the exact single SGD step live in JS (on the current drawing + label + loaded weights) and shows 5 panels — (A) before→after probabilities, (B) per-output-neuron gradient size, (C) the top-5 weight updates worked as `new = old − lr×grad`, (D) the fc2 weight-gradient heatmap, (E) **gradient = predicted − actual** (`softmax − one_hot`, = the fc2 bias gradient). The download lives in the modal footer. Preview only — it doesn't mutate the live model. Verified: JS math matches a real PyTorch SGD step (before/after probs, gradient, identity) to ~1e-7.
- **"Show the maths, worked" derivation:** a collapsible section inside that popup walks the full backward pass with **live numbers from the current drawing** — forward pipeline, softmax, `L = −ln p_y`, the 4-step derivation of `∂L/∂z = predicted − actual`, the chain-rule hop to fc2 weights (with a live Panel B→C bridge recovering the hidden activation `a_j = grad/dz`), the worked update, the η=0.30 catastrophic-forgetting note, and a 45-second interview crib. Equations use **native MathML** (no KaTeX/CDN — CSP-safe); styling reuses the app's tokens.

### ✅ Neuron dissection — random init vs trained · `compare_neurons.py` + front-end "Neuron compare"
- **`compare_neurons.py`** (matplotlib) contrasts a single hidden neuron **before and after training** — the "static → structure" story made visible and measurable. Default neuron **2**, optional CLI index (`python compare_neurons.py 7`).
- Builds two `SimpleNN`s — one freshly random-initialised, one loaded from `model_weights.pth` — and pulls the neuron's 784-weight `fc1` row + bias from each.
- **Receptive fields:** both plotted as 28×28 maps on **one shared** red→green (`RdYlGn`) scale (common max-|weight|), so the trained weights read as genuinely stronger, not just tidier. Each panel's title shows its bias.
- **Metrics** (printed table + on-figure text): weight mean/std, **spatial smoothness** = average Pearson correlation between each weight and its 4 grid neighbours (untrained ≈ 0, trained clearly positive — the number that captures "static → structure"), and bias. Verified run on neuron 2: smoothness **+0.024 → +0.627**.
- **Weights → behaviour:** runs the full 10k MNIST test set through both models, records this neuron's **post-ReLU** activation for every image, and bar-charts its **mean activation per true digit class** (untrained ≈ flat, trained selective). Neuron 2 fires most for digits **6, 8, 2**.
- Saves `neuron_<idx>_compare.png` and prints a plain-language summary (smoothness rise, bias shift, top digits). **Verified working.**
- **Front-end mirror:** `nn_visualizer.html` gains a **"Neuron compare"** card that appears when you click a hidden neuron — the same neuron's receptive field **random-vs-trained** on one shared scale, plus an in-browser mean/std/**smoothness**/bias table (smoothness math verified identical to the Python: 0.627 for trained h[2]). Uses the already-embedded `window.TRAINED` + a fixed-seed random snapshot, so no re-embed is needed; the per-digit selectivity bars stay a Python/PNG deliverable (the browser has no test set).

### 📄 Documentation
- `CLAUDE.md` — guidance for future Claude Code sessions (commands, architecture, conventions, what's intentionally unbuilt).
- `Question&Finding.md` — personal Q&A review notes.

---

- **Step 4 — Evaluation:** load `model_weights.pth` and measure accuracy on the held-out 10k test set (checks it generalises, not just memorises).

---

## How to run

```bash
pip install torch torchvision     # one-time setup
python load_mnist_data.py         # download data + sanity check
python loss_function.py           # Step 3a: see cross-entropy loss react to good vs bad guesses
python optimizer.py               # Step 3b: watch Adam drive the loss down on one batch (0/8 -> 8/8)
python train.py                   # Step 3c: train 5 epochs on 60k images, save model_weights.pth (~98% train acc)
python train_one.py <file.json>   # fine-tune on one exported drawing (one gentle nudge); defaults to newest instance*.json
python embed_weights.py           # push model_weights.pth into nn_visualizer.html, then redeploy the artifact
python compare_neurons.py [idx]   # contrast one hidden neuron (default 2) random-vs-trained -> neuron_<idx>_compare.png (needs matplotlib)
python -c "import torch; from model import SimpleNN; print(SimpleNN()(torch.randn(4,1,28,28)).shape)"   # check the model → [4, 10]
```

**Environment:** Python 3.11, PyTorch 2.x, Windows.
