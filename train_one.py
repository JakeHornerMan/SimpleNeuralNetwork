import sys                                    # for reading a filename argument and clean error exits
import glob                                    # to auto-find the newest exported instance if no file is given
import json                                    # the exported drawing is a small JSON file
import os                                      # to check model_weights.pth exists
import torch                                  # core PyTorch — tensors and autograd
import torch.nn as nn                          # loss function
import torch.optim as optim                    # SGD lives here
from model import SimpleNN                     # our 784 -> 128 -> 10 network

sys.stdout.reconfigure(encoding="utf-8")      # so the ↑ arrow prints instead of crashing on Windows consoles

# ==================================================================
# The SINGULAR TRAINING PATH — correct the model on ONE hand-drawn example.
# ------------------------------------------------------------------
# You drew a digit in the app, the model got it wrong, you picked the true label
# and exported it. This script loads the existing trained model, applies ONE gentle
# nudge toward the right answer on just that drawing, and saves the updated weights.
#
# HONEST CAVEAT (worth understanding): training on a single example is overfitting by
# definition. One gentle step barely moves a 100,000-weight network, but repeat it hard
# on one image and the net starts to "catastrophically forget" other digits — it gets
# better at your drawing while quietly getting worse at everything else. The real way to
# improve a model is to ADD examples to the dataset and retrain (train.py). This path is a
# teaching tool that makes that trade-off visible — use it sparingly.
#
# Reset at any time: delete model_weights.pth and run `python train.py` for a fresh baseline.
# ==================================================================

# LR = step size for the single nudge, and it is DANGEROUS to get wrong. Measured on one misclassified '4'
# (one SGD step, then MNIST test accuracy):
#     LR 0.005 -> corrects it, 0.00% accuracy lost      LR 0.02 -> corrects, ~3% lost
#     LR 0.01  -> corrects it, ~0.3% accuracy lost       LR 0.30 -> corrects, but accuracy COLLAPSES to ~10% (destroyed)
# So one big step doesn't just overfit — it can wipe the whole model ("catastrophic forgetting"). 0.01 is the
# gentle default: reliably fixes the drawing with negligible damage. Nudges are additive — run again for more.
LR = 0.01

# --- locate the exported instance file ---
path = sys.argv[1] if len(sys.argv) > 1 else None        # optional: `python train_one.py myfile.json`
if path is None:                                          # otherwise grab the newest instance*.json in this folder
    matches = sorted(glob.glob('instance*.json'), key=os.path.getmtime)
    if not matches:
        sys.exit("No instance file found. Export a drawing from the app first "
                 "(or pass a path: python train_one.py <file.json>).")
    path = matches[-1]

if not os.path.exists('model_weights.pth'):               # we refine an EXISTING model, so it must be there
    sys.exit("model_weights.pth not found. Run `python train.py` first to create the baseline.")

# --- load the drawing the app exported ---
with open(path) as f:
    inst = json.load(f)
pixels = inst['pixels']                        # 784 floats (0–1) — the exact normalized image the app fed the network
label = int(inst['label'])                     # the true digit you selected under "Actual digit"
x = torch.tensor(pixels, dtype=torch.float32).reshape(1, 1, 28, 28)   # shape the net expects: [batch, channel, 28, 28]
y = torch.tensor([label])                      # the target as a 1-element batch

print(f"Instance: {path}   (true label = {label})")

# --- load the model we're refining ---
model = SimpleNN()                             # empty brain...
model.load_state_dict(torch.load('model_weights.pth'))   # ...poured full of the previously-learned weights
criterion = nn.CrossEntropyLoss()             # same "how wrong" measure as everywhere else
optimizer = optim.SGD(model.parameters(), lr=LR)   # plain SGD keeps a single step predictable and easy to reason about

# --- measure BEFORE, so we can see what the nudge did ---
def snapshot():                                # returns (predicted_digit, prob_on_true_label, loss) without training
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        loss = criterion(logits, y).item()
    return int(probs.argmax()), float(probs[label]), loss

pred0, p0, loss0 = snapshot()

# --- THE ONE GENTLE NUDGE: the standard five steps, exactly once ---
optimizer.zero_grad()                          # 1) clear any old gradients
logits = model(x)                              # 2) forward pass on this one drawing
loss = criterion(logits, y)                    # 3) how wrong it is on the true label
loss.backward()                                # 4) backprop: the slope for every weight
optimizer.step()                               # 5) nudge every weight one small step toward getting THIS drawing right

# --- measure AFTER ---
pred1, p1, loss1 = snapshot()

print(f"  prediction:  {pred0}  ->  {pred1}     (want {label})")
print(f"  p(true={label}): {p0:.3f}  ->  {p1:.3f}     (↑ means more confident in the right answer)")
print(f"  loss:        {loss0:.4f}  ->  {loss1:.4f}")

# --- save the refined weights back over the same file (this is the 'permanent' part) ---
torch.save(model.state_dict(), 'model_weights.pth')
print("Saved updated weights to model_weights.pth")
print("Run `python embed_weights.py` to push these into the visualizer, then redeploy it.")
