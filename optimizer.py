import sys                                     # to force UTF-8 output so the ▼ down-arrows print on Windows consoles
import torch                                  # core PyTorch — tensors, autograd (the machinery that computes gradients)
import torch.nn as nn                          # neural-network building blocks; here we want the loss function
import torch.optim as optim                    # the optimisers live here — Adam, SGD, etc.
from torchvision import datasets, transforms  # to grab a small batch of real digits to practise on
from model import SimpleNN                     # our 784 -> 128 -> 10 network from Step 2

sys.stdout.reconfigure(encoding="utf-8")      # so the ▼ symbols below render instead of turning into mojibake

# ==================================================================
# Step 3b — the optimiser: "how do we FIX the weights?"
# ------------------------------------------------------------------
# The whole game: the network has 100,000+ weights (784*128 + 128*10, plus
# biases). Each is just a number. Training = searching for the combination of
# numbers that makes the network good at digits.
#
# The landscape picture: think of the loss as hilly terrain. Our current weights
# are a position on it; the height there is how wrong we are. High ground = bad,
# low ground = good. Training is walking downhill to the lowest valley.
#
# We can't see the whole 100,000-dimensional landscape, so we can't jump to the
# bottom. We can only feel the slope where we're standing and take one small step
# downhill. Then feel again, step again — thousands of times.
#
# The mechanical loop each step:
#   1. forward pass      -> the network's current guess
#   2. loss              -> how wrong that guess is (the height on the landscape)
#   3. loss.backward()   -> BACKPROP: the slope (gradient) at our position; for
#                           every weight it answers "nudge this up — does error go
#                           up or down, and how steeply?"
#   4. optimiser.step()  -> take the step: move every weight a tiny bit downhill
# ==================================================================

# --- ingredients ---
transform = transforms.ToTensor()             # 0–1 pixel scaling, same as when we loaded the data
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # cached dataset

model = SimpleNN()                            # the network whose 100,000+ weights we're about to adjust
criterion = nn.CrossEntropyLoss()             # the "how wrong" measure from Step 3a (expects raw logits)

# THE OPTIMISER — Adam.
#   * model.parameters() hands Adam every weight and bias in the network — the full
#     100,000+ numbers it is allowed to adjust.
#   * lr = 0.001 is the LEARNING RATE: the step size. Small = careful little steps.
#     Too big and we overshoot the valley (leap over it, bounce around, never settle);
#     too small and training crawls. It's the single most important knob to tune.
#   * Why Adam and not plain gradient descent? Plain SGD uses ONE step size for every
#     weight, everywhere. Adam is ADAPTIVE: it keeps a per-weight step size AND builds
#     up "momentum" — like a ball rolling downhill, gathering speed on consistent
#     slopes and slowing over bumpy ground. Net effect: it reaches a good valley
#     faster and more reliably, which is why it's the default choice.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- grab ONE small batch to practise on ---
# We deliberately reuse the SAME 8 images every step. If the optimiser works, the loss
# on this fixed batch must fall step after step — that's the network memorising 8 digits,
# which is the cleanest possible proof that "nudge the weights" actually reduces error.
images = torch.stack([train_data[i][0] for i in range(8)])          # shape [8, 1, 28, 28]
labels = torch.tensor([train_data[i][1] for i in range(8)])          # the 8 true digits

print("Walking downhill on one fixed batch of 8 digits:")
print(f"True labels: {labels.tolist()}\n")

prev_loss = None                              # remember last step's loss so we can show the SIZE of each downhill step
noted = False                                 # so the "correct but not confident" note prints only once

for step in range(1, 21):                     # take 20 small downhill steps
    optimizer.zero_grad()                     # 1) clear last step's gradients — PyTorch ACCUMULATES them, so
                                              #    without this the slopes would pile up and point the wrong way
    logits = model(images)                    # 2) forward pass -> 10 raw scores per image, shape [8, 10]
    loss = criterion(logits, labels)          # 3) how wrong we are right now = our height on the landscape
    loss.backward()                           # 4) BACKPROP: fill every weight's .grad with the slope at this spot
    optimizer.step()                          # 5) THE NUDGE: Adam moves every weight one small step downhill

    # --- metrics that make the descent legible ---
    probs = torch.softmax(logits, dim=1)                          # the 10 scores turned into probabilities, per image
    p_true = probs.gather(1, labels.unsqueeze(1)).mean().item()   # average probability the net put on the CORRECT digit
    correct = (logits.argmax(dim=1) == labels).sum().item()      # how many of the 8 it currently gets right
    drop = "" if prev_loss is None else f" (▼{prev_loss - loss.item():.3f})"   # how far we descended since the last step

    # loss falls, the per-step drop (▼) shrinks as the slope flattens, and p(true) climbs from a guess toward conviction
    print(f"step {step:2d}  |  loss {loss.item():.4f}{drop}  |  p(true) {p_true:.2f}  |  {correct}/8 correct")

    if correct == 8 and not noted:            # the teaching moment: "correct" arrives long before "confident"
        print("           ^ all 8 already correct — but p(true) is still low; watch the loss keep falling (correct != confident)")
        noted = True

    prev_loss = loss.item()                   # carry this step's loss forward to size the next step's drop

print("\nThe per-step drop (▼) shrinks as we near the valley floor — the slope is flattening out.")
print("And p(true) climbs from ~0.1 (a guess) toward conviction: loss = -ln(p(true)), the exact link back to loss_function.py.")
print("Real training (Step 3c, train.py) does this over the whole 60,000-image set, in fresh batches, for several passes.")
