import torch                                        # core PyTorch library — tensors, autograd, the whole engine
from torchvision import datasets, transforms        # datasets = ready-made data loaders (MNIST etc.); transforms = image preprocessing tools

# Convert each image to a tensor (numbers PyTorch can work with)
transform = transforms.ToTensor()                    # build a preprocessing step: PIL image -> tensor, and rescale pixels from 0–255 to 0.0–1.0

# Fetch MNIST — downloads once, then caches in ./data
train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)  # 60k training set; root=where to store, train=True=training split, download=True=grab it if missing, transform=apply ToTensor to each image
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # 10k test set; train=False selects the held-out split you'll measure accuracy on later

# Sanity check — prove it loaded
print(f"Training images: {len(train_data)}")         # len() on the dataset = number of samples; expect 60000
print(f"Test images:     {len(test_data)}")          # same for the test split; expect 10000

image, label = train_data[0]                          # indexing the dataset returns one (image, label) pair; here the very first sample
print(f"Image shape: {image.shape}")                 # tensor dimensions; expect torch.Size([1, 28, 28]) = 1 channel (grayscale) x 28 x 28 pixels
print(f"First label: {label}")                        # the ground-truth digit (0–9) that image 0 represents
