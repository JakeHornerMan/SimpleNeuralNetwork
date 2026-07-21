import sys                                    # to read the optional neuron-index argument and exit cleanly on errors
import os                                      # to check model_weights.pth exists before we try to load it
import numpy as np                             # for the weight arithmetic (mean/std, the neighbour-correlation metric)
import torch                                  # core PyTorch — we run real forward passes to measure neuron behaviour
from torch.utils.data import DataLoader        # feeds the 10k test images to the model in batches
from torchvision import datasets, transforms  # the MNIST test set + the pixel-to-tensor transform
import matplotlib                              # plotting; we set a non-interactive backend so it saves a file without a window
matplotlib.use("Agg")                          # "Agg" = write straight to a PNG, no GUI needed (works headless / on Windows)
import matplotlib.pyplot as plt                # the actual plotting API
from model import SimpleNN                      # our 784 -> 128 -> 10 network from Step 2

sys.stdout.reconfigure(encoding="utf-8")      # so the arrows/× print instead of crashing on a Windows console

# ==================================================================
# compare_neurons.py — what does TRAINING actually do to a single hidden neuron?
# ------------------------------------------------------------------
# A freshly-initialised network fills every weight with RANDOM noise. That randomness is
# the whole point of the "before" picture: a random neuron has NO SPATIAL STRUCTURE — each
# of its 784 weights is drawn independently, so neighbouring pixels are uncorrelated and the
# 28×28 receptive field looks like TV static. It detects nothing in particular.
#
# Training changes that. Gradient descent slowly CORRELATES the weights: pixels that tend to
# be bright together in the digits this neuron helps classify get pushed the same direction,
# so nearby weights start to agree. The static resolves into a smooth stroke/blob — a real
# FEATURE DETECTOR. This script makes that "static → structure" transition both VISIBLE
# (side-by-side receptive fields) and MEASURABLE (a spatial-smoothness number), then connects
# the weights to BEHAVIOUR by showing which digits the trained neuron actually fires for.
#
# Run:  python compare_neurons.py        (defaults to hidden neuron 2)
#       python compare_neurons.py 7      (any index 0–127)
# Needs model_weights.pth (run train.py first) and the MNIST test set (auto-downloads).
# ==================================================================

# --- which hidden neuron are we dissecting? ---
HID = 128                                      # the hidden layer has 128 neurons (fc1 maps 784 -> 128)
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2   # default neuron 2; accept an optional CLI arg
if not (0 <= idx < HID):                        # guard: the layer only has neurons 0..127
    sys.exit(f"Neuron index must be between 0 and {HID - 1} (got {idx}).")

# --- build the TWO versions of the model we're contrasting ---
# untrained: a brand-new SimpleNN. Its __init__ fills fc1/fc2 with PyTorch's default random init,
#            so this is literally the "before any learning" network — pure noise weights.
untrained = SimpleNN().eval()                   # .eval() just puts it in inference mode (no dropout/batchnorm here, but tidy)

# trained: the same architecture, but poured full of the weights learned in train.py (~98% train acc).
if not os.path.exists("model_weights.pth"):     # we need the learned numbers to have an "after" to compare against
    sys.exit("model_weights.pth not found. Run `python train.py` first to create the trained weights.")
trained = SimpleNN().eval()                      # empty brain...
trained.load_state_dict(torch.load("model_weights.pth"))   # ...poured full of the learned weights

# --- pull out just THIS neuron's parameters from each model ---
# fc1.weight is [128, 784]: row j is neuron j's 784 incoming weights (one per input pixel). fc1.bias is [128].
# .detach().numpy() = drop autograd tracking and hand us a plain NumPy array we can do stats on.
w_untr = untrained.fc1.weight[idx].detach().numpy()   # 784 random weights for our neuron
w_trn = trained.fc1.weight[idx].detach().numpy()      # 784 learned weights for our neuron
b_untr = float(untrained.fc1.bias[idx].detach())       # this neuron's random bias (.detach() = drop autograd tracking)
b_trn = float(trained.fc1.bias[idx].detach())          # this neuron's learned bias
map_untr = w_untr.reshape(28, 28)               # lay the 784 weights back over the input image -> the receptive field
map_trn = w_trn.reshape(28, 28)                 # same for the trained neuron


# --- the headline metric: SPATIAL SMOOTHNESS = do neighbouring weights agree? ---
def neighbour_correlation(field):
    """Average Pearson correlation between each weight and its immediate grid neighbours.

    We collect every adjacent pair on the 28×28 grid — each cell with the one to its RIGHT and
    the one BELOW it (that covers all 4-neighbour relationships across the whole grid, each edge
    counted once) — then correlate the 'this cell' list against the 'neighbour' list.
      random weights  -> neighbours are independent -> correlation ~ 0  (static)
      trained weights -> neighbours move together    -> correlation > 0  (smooth structure)
    """
    left = np.concatenate([field[:, :-1].ravel(),   # each cell paired with its RIGHT neighbour
                           field[:-1, :].ravel()])   # ...and each cell paired with the one BELOW it
    right = np.concatenate([field[:, 1:].ravel(),    # the corresponding right-neighbours
                            field[1:, :].ravel()])    # ...and the corresponding below-neighbours
    return float(np.corrcoef(left, right)[0, 1])     # Pearson r between the two aligned lists

smooth_untr = neighbour_correlation(map_untr)   # expect ~0 for the random neuron
smooth_trn = neighbour_correlation(map_trn)     # expect clearly positive for the trained neuron


# --- connect weights to BEHAVIOUR: how does each version fire across the real test digits? ---
# Load the held-out 10k test set exactly like the other scripts (train=False = the images the net never trained on).
transform = transforms.ToTensor()               # 28×28 image -> tensor, pixels scaled to 0.0–1.0
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)   # the 10k test set (cached)
test_loader = DataLoader(test_data, batch_size=256)   # batches of 256 — no shuffle needed, we just read activations


def per_digit_mean_activation(model):
    """Run every test image through `model` and return this neuron's mean POST-ReLU activation per digit class.

    Post-ReLU = the value the neuron actually passes on (relu(fc1(x))[idx]); negatives are clamped to 0, so this
    is 'how hard the neuron fires'. We average that firing over all test images of each true label 0–9. A neuron
    that has learned a feature will fire harder for the digits containing that feature -> an uneven bar chart.
    """
    sums = np.zeros(10)                          # running total of the neuron's activation, per digit class
    counts = np.zeros(10)                        # how many test images we've seen of each class (for the average)
    with torch.no_grad():                        # inference only — no gradients, faster and lighter
        for images, labels in test_loader:       # one batch of images + their true labels at a time
            flat = model.flatten(images)         # [B, 1, 28, 28] -> [B, 784]
            acts = model.relu(model.fc1(flat))[:, idx].numpy()   # push through fc1 + ReLU, keep only OUR neuron's column
            for a, y in zip(acts, labels.numpy()):
                sums[y] += a                     # add this image's activation to its true-digit bucket
                counts[y] += 1                   # count it, so we can divide for the mean
    return sums / counts                         # mean activation for digits 0..9

means_untr = per_digit_mean_activation(untrained)   # flat/random — the untrained neuron has no preference
means_trn = per_digit_mean_activation(trained)      # uneven — the trained neuron favours certain digits


# ================= draw the comparison figure =================
# Layout: top row = the two receptive fields (shared colour scale) + a shared colour bar;
#         bottom row (full width) = the per-digit mean-activation bars, untrained vs trained.
fig = plt.figure(figsize=(11, 9))               # a tall-ish canvas with room for two panels stacked
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.18)
ax_u = fig.add_subplot(gs[0, 0])                # untrained receptive field
ax_t = fig.add_subplot(gs[0, 1])                # trained receptive field
ax_b = fig.add_subplot(gs[1, :])                # the behaviour bar chart, spanning both columns

fig.suptitle(f"Hidden neuron h[{idx}]  ·  random init  →  trained", fontsize=15, fontweight="bold")

# SHARED colour scale so the two maps are DIRECTLY comparable: one symmetric range built from the
# largest magnitude across BOTH neurons. RdYlGn = red (negative) → yellow (~0) → green (positive),
# matching the app's weight-inspector semantics. Because the scale is shared, the random neuron's
# small, evenly-spread weights look pale and noisy while the trained neuron's strong weights pop.
M = float(max(np.abs(map_untr).max(), np.abs(map_trn).max()))   # common max-|weight| across both maps
im = ax_u.imshow(map_untr, cmap="RdYlGn", vmin=-M, vmax=M)      # note: same vmin/vmax used for both panels
ax_t.imshow(map_trn, cmap="RdYlGn", vmin=-M, vmax=M)            # <-- shared scale is the whole point
ax_u.set_title(f"Untrained (random)\nbias = {b_untr:+.3f}", fontsize=11)
ax_t.set_title(f"Trained\nbias = {b_trn:+.3f}", fontsize=11)
for ax in (ax_u, ax_t):                          # these are images, not plots — hide the pixel-index ticks
    ax.set_xticks([]); ax.set_yticks([])
# one colour bar shared by both maps, so the reader knows both panels use the same red↔green meaning
cbar = fig.colorbar(im, ax=[ax_u, ax_t], fraction=0.046, pad=0.04)
cbar.set_label("weight value  (red − · green +)", fontsize=9)

# --- the behaviour panel: grouped bars, untrained vs trained, one group per digit 0–9 ---
x = np.arange(10)                                # the ten digit classes on the x-axis
width = 0.4                                      # each digit gets two side-by-side bars
ax_b.bar(x - width/2, means_untr, width, label="untrained", color="#9aa4b2")   # grey = the random baseline
ax_b.bar(x + width/2, means_trn, width, label="trained", color="#2ee08a")      # green = the learned neuron
ax_b.set_xticks(x)                               # label every digit 0..9
ax_b.set_xlabel("true digit class")
ax_b.set_ylabel("mean activation (post-ReLU)")
ax_b.set_title("What makes this neuron fire?  Mean activation per digit — "
               "flat = no preference, uneven = a learned feature", fontsize=11)
ax_b.legend()
ax_b.grid(axis="y", alpha=0.25)                  # faint horizontal grid so bar heights are easy to read

# --- the metrics table, printed ON the figure so the picture is self-explanatory ---
# Monospace so the columns line up. Smoothness is the star: near-zero for random, clearly positive for trained.
tbl = (f"                 untrained     trained\n"
       f"weight mean     {w_untr.mean():+9.4f}   {w_trn.mean():+9.4f}\n"
       f"weight std      {w_untr.std():9.4f}   {w_trn.std():9.4f}\n"
       f"smoothness      {smooth_untr:+9.4f}   {smooth_trn:+9.4f}   <- neighbour correlation (static → structure)\n"
       f"bias            {b_untr:+9.4f}   {b_trn:+9.4f}")
fig.text(0.5, 0.005, tbl, ha="center", va="bottom", family="monospace", fontsize=9,
         bbox=dict(boxstyle="round", facecolor="#f2f4f7", edgecolor="#c8cdd6"))

out_path = f"neuron_{idx}_compare.png"           # the deliverable image, named for the neuron we dissected
fig.savefig(out_path, dpi=130, bbox_inches="tight")   # write the PNG at a readable resolution
plt.close(fig)


# ================= plain-language summary to stdout =================
top3 = np.argsort(means_trn)[::-1][:3]           # the three digits the TRAINED neuron fires hardest for
print(f"\nNeuron h[{idx}] — before vs after training")
print("-" * 52)
print(f"Spatial smoothness (neighbour correlation): {smooth_untr:+.3f} -> {smooth_trn:+.3f}")
print(f"  The random neuron's weights are ~uncorrelated static; training raised the")
print(f"  neighbour agreement by {smooth_trn - smooth_untr:+.3f}, resolving it into structure.")
print(f"Bias: {b_untr:+.3f} -> {b_trn:+.3f}  (shift {b_trn - b_untr:+.3f}) — "
      f"training re-tuned how easily this neuron fires.")
print(f"After training this neuron responds MOST to digits: "
      f"{top3[0]}, {top3[1]}, {top3[2]}  (highest mean activation).")
print(f"  So its receptive field is the visual feature those digits tend to share.")
print(f"\nSaved figure to {out_path}")
