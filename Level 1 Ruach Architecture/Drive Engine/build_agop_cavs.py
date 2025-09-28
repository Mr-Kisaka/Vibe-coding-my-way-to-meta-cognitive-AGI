import json
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import os

# === CONFIG ===
MODEL_NAME = ""  # Update Model name
INPUT_FILE = ""  # Updated filename
ACTIVATION_FILE = "[affect]_layerwise_activations.npz"  # Replace [affect]
OUT_FILE = "[affect]_agop_cavs.npz"  # Replace [affect]
LAYERS = list(range( 0, ))  # Specify layer range

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()

#Replace "affect+" and "affect-" throughout the document with actual emotion
# === Load dataset ===
print(f"Loading affect+/affect- dataset: {INPUT_FILE}")
samples = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    # ==== Load Dataset ====
 print(f"Loading affect+/affect- dataset: {INPUT_FILE}")
samples = []

label_map = {
    "affect+": 1,   # Positive class
    "affect-": 0       # Negative class
}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        affect = j["affect"].lower().strip()
        if affect in label_map:
            samples.append((j["text"], label_map[affect]))
        else:
            print(f"Unknown affect: {j['affect']}, skipping")

            continue

print(f"{len(samples)} total samples")

# Count distribution
affect+_count = sum(1 for _, label in samples if label == 1)
affect-_count = sum(1 for _, label in samples if label == 0)
print(f"affect+ samples: {affect+_count}, affect- samples: {affect-_count}") 

# === Step 1: Extract layerwise activations (or load cached)
layer_data = {layer: {"X": [], "y": []} for layer in LAYERS}

if os.path.exists(ACTIVATION_FILE):
    print(f"Loading cached activations from {ACTIVATION_FILE}")
    cache = np.load(ACTIVATION_FILE)
    for layer in LAYERS:
        layer_data[layer]["X"] = cache[f"layer_{layer}_X"]
        layer_data[layer]["y"] = cache[f"layer_{layer}_y"]
else:
    print(f"Extracting activations from layers {LAYERS[0]}–{LAYERS[-1]}")

    def extract_layerwise(text):
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.transformer(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]  # skip embedding layer
            reps = [h[0].mean(dim=0).cpu().numpy() for h in hidden_states]
            return reps

    for text, label in tqdm(samples, desc="Extracting"):
        try:
            reps = extract_layerwise(text)
            for i in LAYERS:
                layer_data[i]["X"].append(reps[i])
                layer_data[i]["y"].append(label)
        except Exception as e:
            print(f"Skipped: {e}")

    print(f"Caching activations to {ACTIVATION_FILE}")
    to_save = {}
    for layer in LAYERS:
        to_save[f"layer_{layer}_X"] = np.stack(layer_data[layer]["X"])
        to_save[f"layer_{layer}_y"] = np.array(layer_data[layer]["y"])
    np.savez_compressed(ACTIVATION_FILE, **to_save)

# === Step 2: Train classifier and compute AGOP vectors ===
def compute_agop(X, y):
    X = np.stack(X)
    clf = LogisticRegression(solver="liblinear", max_iter=1000).fit(X, y)
    weight = torch.tensor(clf.coef_[0], dtype=torch.float32, requires_grad=False)
    bias = torch.tensor(clf.intercept_[0], dtype=torch.float32, requires_grad=False)

    grads = []
    for i in range(len(X)):
        xi = torch.tensor(X[i], dtype=torch.float32, requires_grad=True)
        logit = torch.dot(xi, weight) + bias
        prob = torch.sigmoid(logit)
        target = torch.tensor(float(y[i]), dtype=torch.float32).unsqueeze(0)
        loss = F.binary_cross_entropy(prob.unsqueeze(0), target)
        loss.backward()
        grads.append(xi.grad.detach().numpy())

    grads = np.stack(grads)
    agop = grads.T @ grads / len(grads)
    eigvals, eigvecs = np.linalg.eigh(agop)
    return eigvecs[:, -1]  # top eigenvector

print("Computing AGOP vectors...")
agop_cavs = {}
for i in tqdm(LAYERS, desc="🔢 Layers"):
    X = layer_data[i]["X"]
    y = layer_data[i]["y"]
    if len(X) < 10:
        print(f"Skipping layer {i} (too few samples)")
        continue
    cav = compute_agop(X, y)
    agop_cavs[f"affect+_layer_{i}"] = cav.astype(np.float32)

# === Save AGOP CAVs
print(f"Saving AGOP CAVs to {OUT_FILE}")
np.savez_compressed(OUT_FILE, **agop_cavs)
print("Done.")
