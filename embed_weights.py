import re                                      # to find/replace the embedded data block
import json                                    # to serialize the base64 payload as JS
import base64                                  # weights are embedded as base64 so they can live inline in the HTML
import sys                                     # clean error exit
import os
import torch                                  # to load the saved model_weights.pth

# ==================================================================
# Push the current model_weights.pth INTO the visualizer.
# ------------------------------------------------------------------
# The web app can't read model_weights.pth (browser sandbox), so the trained numbers are
# embedded directly in nn_visualizer.html as a base64 blob (window.TRAINED). After you refine
# the model (train.py or train_one.py), run this to refresh that blob, then redeploy the artifact.
#
# The byte layout must match how the browser's forward pass indexes the weights:
#   - little-endian float32 (JS Float32Array is little-endian on browsers)
#   - row-major, matching W[j*IN + i] — which is exactly PyTorch's nn.Linear [out, in] layout.
# ==================================================================

HTML = 'nn_visualizer.html'
PTH = 'model_weights.pth'

if not os.path.exists(PTH):
    sys.exit(f"{PTH} not found. Run `python train.py` first.")

sd = torch.load(PTH)

def enc(t):                                    # tensor -> base64 of its little-endian float32 bytes (row-major)
    return base64.b64encode(t.detach().cpu().numpy().astype('<f4').tobytes()).decode('ascii')

data = {
    'fc1w': enc(sd['fc1.weight']),             # [128, 784]
    'fc1b': enc(sd['fc1.bias']),               # [128]
    'fc2w': enc(sd['fc2.weight']),             # [10, 128]
    'fc2b': enc(sd['fc2.bias']),               # [10]
}

script = '<script id="trained-data">window.TRAINED=' + json.dumps(data) + ';</script>\n'

html = open(HTML, encoding='utf-8').read()
# idempotent: remove any previous injection before adding the fresh one
html = re.sub(r'<script id="trained-data">.*?</script>\n', '', html, flags=re.S)
marker = '<script>\n(() => {'                   # the main app script — we inject the data right before it
if marker not in html:
    sys.exit("Could not find the main <script> marker in the HTML; aborting so nothing is corrupted.")
html = html.replace(marker, script + marker, 1)
open(HTML, 'w', encoding='utf-8').write(html)

kb = round(sum(len(v) for v in data.values()) / 1024, 1)
print(f"Embedded {kb} KB of weights into {HTML}.")
print("Now redeploy the artifact to see the refined model live.")
