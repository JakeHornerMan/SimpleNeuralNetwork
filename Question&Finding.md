# Questions & Findings

A personal review log — questions I asked while building, and the answers.

---

### Q: If I run `train.py` again, does the network get stronger?

Not run-over-run. `train.py` starts with `model = SimpleNN()` (random weights) every time, trains 5 fresh epochs, and **overwrites** `model_weights.pth`. So each run is a **restart**, not a continuation — another ~98% run, but it doesn't stack.

Two ways to actually make it stronger:
- **Train longer in one run** — raise `EPOCHS` (e.g. 5 → 10). Diminishing returns fast, and eventually **overfitting**: memorising the training set without improving on unseen digits.
- **Resume from the saved file** — `model.load_state_dict(...)` *before* the training loop so it picks up where it left off instead of starting random. This is the "keep getting stronger" version.

---

### Q: How is the trained data saved — is it `model_weights.pth`?

Yes. That one file is the entire "brain."

- `model.state_dict()` = a dictionary of the learned numbers. Keys: `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`.
- `torch.save(...)` pickles that dict to disk (~400 KB, ~100k numbers).

**Contains:** the weights and biases (the values training shaped).
**Doesn't contain:** the architecture or any code — the `.pth` is meaningless without `model.py`.

To use it: rebuild the same structure (`SimpleNN()`) and pour the numbers back in with `model.load_state_dict(torch.load('model_weights.pth'))`. Structure comes from the code; learned values come from the file.

**Mental model:** `model.py` = the empty brain, `model_weights.pth` = the memories. Same net, different weights file → different "knowledge."

---

### Q: What is the gradient at the output layer?

For softmax + cross-entropy, the gradient of the loss w.r.t. the output logits is beautifully simple:

**`dL/dlogits = softmax(logits) − one_hot(true_label)`**  — i.e. **predicted − actual**.

- For the **true** class: `p_true − 1` → negative → its logit gets pushed **up**.
- For every **wrong** class: `p_wrong` → positive → its logit gets pushed **down**.
- This vector is *exactly* the output layer's **bias gradient** (`fc2.bias.grad`), and each output weight gradient is just `dz[o] · h[j]`.

Verified numerically (`softmax − one_hot` vs `logits.grad` vs `fc2.bias.grad` all match to ~1e-8). Being able to say "the gradient at the output is just predicted minus actual" shows you understand the maths, not just the API. The visualizer's **"one SGD step, explained"** popup (Export for training) shows this live.
