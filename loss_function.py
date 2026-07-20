import torch                                  # core PyTorch — tensors and math
import torch.nn as nn                          # neural-network building blocks; here we want the loss function
from torchvision import datasets, transforms  # to grab one real MNIST image to score
from model import SimpleNN                     # our 784 -> 128 -> 10 network from Step 2

# ------------------------------------------------------------------
# Step 3a — the loss function: "how wrong was the guess?"
# CrossEntropyLoss turns the 10 raw output scores + the true label
# into ONE number. Small = confident and right. Big = confident and wrong.
# ------------------------------------------------------------------

# The loss function itself. NOTE: it expects RAW LOGITS (not softmax) —
# it applies softmax internally — which is exactly why model.py's forward()
# deliberately returns raw scores. Feeding it softmax'd values would double-count.
criterion = nn.CrossEntropyLoss()             # create the cross-entropy loss object once, reuse it every step

# --- Part 1: score the untrained network on one real digit ---
transform = transforms.ToTensor()             # same 0–1 pixel scaling used when we loaded the data
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # reuse the cached dataset

image, label = train_data[0]                  # first sample: image is [1, 28, 28], label is an int (the digit 5)
batch = image.unsqueeze(0)                    # add a batch dimension -> [1, 1, 28, 28]; the model expects a batch, not one loose image

model = SimpleNN()                            # fresh, UNTRAINED network (random weights)
logits = model(batch)                         # forward pass -> 10 raw scores, shape [1, 10]

# CrossEntropyLoss wants the target as a tensor of class indices (a LongTensor), one per item in the batch.
target = torch.tensor([label])                # wrap the true digit in a 1-element batch tensor, e.g. tensor([5])

loss = criterion(logits, target)             # compare the 10 scores against the true label -> a single loss number

print("=== One real digit through the untrained net ===")
print(f"True label:      {label}")            # the digit the image actually is
print(f"Raw logits:      {logits.detach().numpy().round(2)}")  # the 10 scores; detach() = we only want to read them, not track gradients
print(f"Loss:            {loss.item():.4f}")  # .item() pulls the single number out of the tensor
# For 10 classes, a totally clueless net sits around ln(10) ≈ 2.30 — random guessing. Expect roughly that here.

# --- Part 2: build the intuition with hand-made scores ---
# We hand-write logits so you can SEE the loss shrink as the guess gets better.
# Pretend the true answer is class 3 in every case below.
truth = torch.tensor([3])                     # the correct class for these toy examples

confident_right = torch.tensor([[0., 0., 0., 9., 0., 0., 0., 0., 0., 0.]])  # a huge score on class 3 (the right answer)
confident_wrong = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 9., 0., 0.]])  # a huge score on class 7 (the wrong answer)
unsure          = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])  # all equal — no opinion at all

print("\n=== How the loss reacts (true class = 3) ===")
print(f"Confident & RIGHT: {criterion(confident_right, truth).item():.4f}")  # tiny loss — barely wrong at all
print(f"Unsure (guessing): {criterion(unsure, truth).item():.4f}")           # ~2.30, the ln(10) baseline
print(f"Confident & WRONG: {criterion(confident_wrong, truth).item():.4f}")  # big loss — the whole point of the penalty
