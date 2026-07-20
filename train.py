import torch                                  # core PyTorch — tensors and autograd
import torch.nn as nn                          # loss function lives here
import torch.optim as optim                    # optimisers (Adam) live here
from torch.utils.data import DataLoader        # feeds the dataset to the model in shuffled mini-batches
from torchvision import datasets, transforms  # the MNIST dataset + pixel-to-tensor transform
from model import SimpleNN                     # our 784 -> 128 -> 10 network from Step 2

# ==================================================================
# Step 3c — the training loop: do the "nudge the weights" step for REAL,
# over the whole 60,000-image dataset, many times, then save what it learned.
# ==================================================================

# --- data ---
transform = transforms.ToTensor()             # each 28x28 image -> tensor, pixels scaled to 0.0–1.0
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # the 60k training set (cached)

# The DataLoader hands us the data in mini-batches instead of all 60k at once.
#   batch_size=64 -> the network sees 64 images, takes one downhill step, repeats. Small batches = more
#                    steps per epoch and a bit of helpful noise in the gradient.
#   shuffle=True  -> reshuffle the order every epoch so the network never learns the sequence, only the digits.
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# --- ingredients ---
model = SimpleNN()                            # a fresh, untrained network — the weights we're about to shape
criterion = nn.CrossEntropyLoss()             # "how wrong" measure (expects raw logits; applies softmax internally)
optimizer = optim.Adam(model.parameters(), lr=0.001)   # adaptive gradient descent w/ momentum; lr=0.001 = step size

EPOCHS = 5                                     # one epoch = one full pass over all 60,000 images
print(f"Training for {EPOCHS} epochs on {len(train_data)} images "
      f"({len(train_loader)} batches of 64)...\n")

for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0                         # sum of batch losses this epoch -> divide later for the average
    correct = 0                                # how many images we classified correctly this epoch
    total = 0                                  # how many images we've seen this epoch (for the accuracy denominator)

    for images, labels in train_loader:        # each iteration is ONE mini-batch of 64 images + their true labels
        # --- the standard five steps, once per batch ---
        optimizer.zero_grad()                  # 1) clear last batch's gradients (PyTorch accumulates them)
        logits = model(images)                 # 2) forward pass -> [64, 10] raw scores
        loss = criterion(logits, labels)       # 3) how wrong this batch's guesses are -> one number
        loss.backward()                        # 4) backprop: compute the slope (gradient) for every weight
        optimizer.step()                       # 5) the nudge: Adam moves every weight a small step downhill

        # --- bookkeeping so we can watch loss fall and accuracy rise ---
        running_loss += loss.item()            # accumulate this batch's loss
        preds = logits.argmax(dim=1)           # the predicted digit per image (highest score)
        correct += (preds == labels).sum().item()   # count the ones we got right
        total += labels.size(0)                # add this batch's image count (64, except maybe the last batch)

    avg_loss = running_loss / len(train_loader)   # mean loss per batch across the epoch
    accuracy = 100.0 * correct / total            # training accuracy as a percentage
    print(f"Epoch {epoch}/{EPOCHS}  |  avg loss {avg_loss:.4f}  |  train accuracy {accuracy:.2f}%")

# --- save the learned weights ---
# state_dict() is just the dictionary of all the trained numbers (weights + biases). Saving it means we can
# reload this exact trained network later WITHOUT retraining — e.g. to feed real weights into the visualizer.
torch.save(model.state_dict(), 'model_weights.pth')
print("\nDone. Saved trained weights to model_weights.pth")
