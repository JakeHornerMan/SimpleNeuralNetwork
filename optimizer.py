import torch                                  # core PyTorch — tensors, autograd (the machinery that computes gradients)
import torch.nn as nn                          # neural-network building blocks; here we want the loss function
import torch.optim as optim                    # the optimisers live here — Adam, SGD, etc.
from torchvision import datasets, transforms  # to grab a small batch of real digits to practise on
from model import SimpleNN                     # our 784 -> 128 -> 10 network from Step 2

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

for step in range(1, 21):                     # take 20 small downhill steps
    optimizer.zero_grad()                     # 1) clear last step's gradients — PyTorch ACCUMULATES them, so
                                              #    without this the slopes would pile up and point the wrong way
    logits = model(images)                    # 2) forward pass -> 10 raw scores per image, shape [8, 10]
    loss = criterion(logits, labels)          # 3) how wrong we are right now = our height on the landscape
    loss.backward()                           # 4) BACKPROP: fill every weight's .grad with the slope at this spot
    optimizer.step()                          # 5) THE NUDGE: Adam moves every weight one small step downhill

    if step == 1 or step % 2 == 0:            # print a few checkpoints so we can watch it descend
        preds = logits.argmax(dim=1)          # the network's current guesses (highest score per image)
        correct = (preds == labels).sum().item()   # how many of the 8 it now gets right
        print(f"step {step:2d}  |  loss {loss.item():.4f}  |  {correct}/8 correct")

print("\nLoss falls and the guesses become correct - the weights are being pulled downhill.")
print("Real training (Step 3c) does this over the whole 60,000-image set, in fresh batches, for several passes.")
