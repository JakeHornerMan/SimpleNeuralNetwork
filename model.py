import torch.nn as nn                      # nn = PyTorch's neural-network building blocks (layers, activations, Module base class)


class SimpleNN(nn.Module):                   # our model inherits from nn.Module — the base class every PyTorch model uses
    def __init__(self):
        super().__init__()                   # run nn.Module's setup so it can track our layers and their parameters

        self.flatten = nn.Flatten()          # turn each [1, 28, 28] image into a flat vector of 784 values (28*28) the linear layers can read
        self.fc1 = nn.Linear(784, 128)       # hidden layer: fully-connected, maps 784 inputs -> 128 neurons (learns weighted combinations of pixels)
        self.relu = nn.ReLU()                # activation: replaces negatives with 0, adding non-linearity so the net can learn complex patterns
        self.fc2 = nn.Linear(128, 10)        # output layer: maps 128 hidden features -> 10 scores, one per digit class (0–9)

    def forward(self, x):                    # defines how data flows through the network on each pass
        x = self.flatten(x)                  # [batch, 1, 28, 28] -> [batch, 784]
        x = self.fc1(x)                      # apply hidden layer -> [batch, 128]
        x = self.relu(x)                     # apply ReLU activation elementwise
        x = self.fc2(x)                      # apply output layer -> [batch, 10] raw scores (logits)
        return x                             # return the 10 logits; a loss like CrossEntropyLoss will turn these into class probabilities later
