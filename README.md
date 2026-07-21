# SimpleNeuralNetwork

A learn-by-building project: a feed-forward neural network for **MNIST handwritten-digit classification**, assembled one step at a time in PyTorch. This is a teaching project where understanding is prioritized over brevity — nearly every line is commented to explain the "why" behind the code.

## What You'll Build

```
Input (784) → Linear(784, 128) → ReLU → Linear(128, 10) → Output (10 logits)
```

A simple two-layer neural network that learns to classify handwritten digits with ~98% training accuracy.

## Quick Start

### Setup
```bash
pip install torch torchvision      # One-time setup (also installs numpy, pillow)
python load_mnist_data.py          # Download MNIST (60k train, 10k test) into ./data
```

### Run the Tutorial Steps
```bash
python loss_function.py            # Step 3a: Cross-entropy loss explained
python optimizer.py                # Step 3b: Watch Adam optimizer nudge weights
python train.py                    # Step 3c: Train 5 epochs on 60k images → model_weights.pth
```

### Interactive Visualizer
Open `nn_visualizer.html` in your browser to:
- **Draw a digit** and watch it flow through the network in real-time
- **Toggle trained vs random weights** to see the difference learning makes
- **Click hidden neurons** to inspect their learned weights as 28×28 heatmaps
- **See the math:** click "Export for training" to watch a single SGD step calculated live in JavaScript

### Fine-tune on Your Drawings
```bash
# Export a drawing from the visualizer as instance-<label>.json, then:
python train_one.py instance-<label>.json    # One gentle SGD step (LR=0.01)
python embed_weights.py                       # Push updated weights back into the HTML
```

### Analysis & Comparison
```bash
python compare_neurons.py              # Random vs trained neuron 2
python compare_neurons.py 7            # Random vs trained neuron 7 (optional index)
```

## Project Status

### ✅ Completed Steps

| Step | Script | What It Does |
|------|--------|-------------|
| 1 | `load_mnist_data.py` | Fetch 60k training + 10k test images; rescale pixels 0–255 → 0.0–1.0 |
| 2 | `model.py` | Define `SimpleNN`: Flatten → Linear(784, 128) → ReLU → Linear(128, 10) |
| 3a | `loss_function.py` | Cross-entropy loss: "how wrong was the prediction?" |
| 3b | `optimizer.py` | Adam optimizer: backprop + weight nudging on a single batch |
| 3c | `train.py` | Full training loop: 5 epochs, batch 64, Adam lr=0.001 → **98% train acc** |
| – | `nn_visualizer.html` | Interactive front-end: draw digits, toggle weights, inspect neurons, see the math |
| 3.5 | `train_one.py` + `embed_weights.py` | Fine-tune on single exported drawings (one SGD step per run) |
| – | `compare_neurons.py` | Visual proof that training adds structure: random ≈ static, trained ≈ feature detectors |

### 📋 Remaining

- **Step 4 — Evaluation:** Measure held-out test-set accuracy (training acc ≠ generalization)

## Key Discoveries

### The Loss Function
`CrossEntropyLoss` takes raw logits (10 numbers) + true label and returns one number:
- Small (~0.01) when confident and right
- Medium (~2.30) when random/uncertain
- Large (~10) when confident and wrong

The baseline "random guess" loss is `ln(10) ≈ 2.30`.

### Training Collapse
Fine-tuning is dangerous: one SGD step at **LR=0.30 collapsed** test accuracy from 97.5% → ~10%. The safe default is **LR=0.01**, which recovers misclassified examples with only ~0.3% accuracy cost.

### Spatial Structure in Weights
Random initialization: neuron weights look like noise (no spatial smoothness).
After training: weights form coherent patterns — **spatial smoothness jumps from ~0.0 → ~0.63**.

This is measured as the average Pearson correlation between each weight and its 4 grid neighbours. It's the visual proof that neurons become feature detectors.

## Architecture Details

### `model.py` — The Network
```python
SimpleNN(nn.Module):
  Flatten(1, 28, 28) → (784)
  Linear(784, 128)
  ReLU
  Linear(128, 10)  # Returns logits (no softmax — CrossEntropyLoss applies it)
```

**Critical:** `forward()` returns raw logits, not probabilities. This is intentional because `nn.CrossEntropyLoss` applies softmax internally.

### `train.py` — The Training Loop
- **DataLoader:** batch_size=64, shuffle=True
- **Loss:** `nn.CrossEntropyLoss`
- **Optimizer:** `torch.optim.Adam(lr=0.001)`
- **Duration:** 5 epochs over 60k images
- **Result:** model_weights.pth saved after training

### `nn_visualizer.html` — The Front-End
A self-contained HTML/JS artifact that:
1. **Reimplements the forward pass in JavaScript** — no server calls, CSP-safe
2. **Preprocesses drawings MNIST-style** — crop → scale to 20px → centre by centre-of-mass
3. **Loads trained weights** as base64-embedded data (~530 KB)
4. **Shows the math** — cross-entropy loss, SGD step derivation, gradient flow (all interactive, using native MathML)
5. **Inspects neurons** — click any neuron to see its 784 weights as a 28×28 heatmap

### `compare_neurons.py` — Teaching Dissection
Contrasts one hidden neuron before and after training:
- **Receptive field:** 28×28 weight heatmap (red=negative, green=positive)
- **Metrics:** mean, std, **spatial smoothness**, bias
- **Behavior:** which digit classes does this neuron fire on? (uses 10k test set)
- **Output:** `neuron_<idx>_compare.png` + printed summary

## Environment

- **Python:** 3.11+
- **PyTorch:** 2.x
- **Dependencies:** torch, torchvision (also installs numpy, pillow)
- **Optional:** matplotlib (needed for `compare_neurons.py`)
- **Platform:** Tested on Windows; should work on macOS/Linux

## File Structure

```
SimpleNeuralNetwork/
├── README.md                    # This file
├── CLAUDE.md                    # Guidance for Claude Code sessions
├── OverviewPrompt.md            # Detailed project progress notes
├── load_mnist_data.py           # Step 1: Download MNIST
├── model.py                     # Step 2: Define the network
├── loss_function.py             # Step 3a: Cross-entropy loss demo
├── optimizer.py                 # Step 3b: Adam + backprop demo
├── train.py                     # Step 3c: Full training loop
├── train_one.py                 # Fine-tune on one exported drawing
├── embed_weights.py             # Embed model_weights.pth into HTML
├── compare_neurons.py           # Random vs trained neuron dissection
├── nn_visualizer.html           # Interactive front-end (live Artifact)
├── model_weights.pth            # Trained weights (generated after train.py)
├── data/                        # MNIST dataset (auto-generated)
└── instance-*.json              # Exported drawings for fine-tuning
```

## Running Validation Checks

Check that `model.py` instantiates correctly:
```bash
python -c "import torch; from model import SimpleNN; print(SimpleNN()(torch.randn(4,1,28,28)).shape)"
# Expected output: torch.Size([4, 10])
```

Check the training result:
```bash
python train.py
# Look for final epoch: ~97–98% accuracy, ~0.067 avg loss
```

Check the visualizer:
1. Open `nn_visualizer.html` in your browser
2. Draw a digit
3. Click "Load trained weights" and draw again — predictions should be ~99% accurate
4. Click a hidden neuron to see its learned feature detector

## Notes for Future Development

- **Don't remove logits:** The `model.py` forward pass intentionally omits softmax. CrossEntropyLoss expects raw logits.
- **Watch the learning rate:** Fine-tuning uses LR=0.01 as a safe default. Higher values (LR=0.30+) cause catastrophic forgetting.
- **Re-embed weights:** If you change the model architecture, you must:
  1. Update the JS forward pass in `nn_visualizer.html`
  2. Update the network diagram layout
  3. Re-run `embed_weights.py`
- **Test-set accuracy:** Step 4 (evaluation) should measure generalization on the held-out 10k test set, not just training accuracy.

## Learning Goals

By the end of this project, you'll understand:
- How to load and preprocess real data (MNIST)
- How to build and initialize a neural network
- How cross-entropy loss quantifies "wrongness"
- How backpropagation computes gradients
- How adaptive optimizers (Adam) descend a loss landscape
- How to train a network end-to-end and save weights
- How to fine-tune on single instances
- How to visualize learned features and verify they make sense

---

**Status:** This is an active learning project. Check `CLAUDE.md` for commands and conventions, and `OverviewPrompt.md` for detailed progress notes.
