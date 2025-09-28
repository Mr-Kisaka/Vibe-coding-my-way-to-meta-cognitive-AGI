# generate_with_fear.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Configuration
MODEL_NAME = "" # Update model name
CAV_FILE = r""        #Input file path
LAYERS = list(range(5, ))  # Later layers for stronger affect
ALPHA = 12  # Steering strength
NUM_SAMPLES = 3  
MAX_NEW_TOKENS = 200  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 50
TOP_P = 0.95
TEMPERATURE = 1.0  

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print(f"Model loaded on {DEVICE}")

# Load and normalize CAVs
print(f"Loading AGOP vectors from {CAV_FILE}")
try:
    cavs = np.load(CAV_FILE)
    cav_dict = {}
    for i in LAYERS:
        vec = cavs[f"affect+_layer_{i}"]
        # Normalize and scale
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm * 2.0  # Adjust scale to liking
        else:
            print(f"Warning: Zero norm for layer {i}, skipping normalization")
        cav_dict[i] = vec
    print(f"Loaded and normalized CAVs for layers: {list(cav_dict.keys())}")
except FileNotFoundError:
    print(f"Error: {CAV_FILE} not found")
    exit(1)
except KeyError as e:
    print(f"Error: Missing CAV for layer in {CAV_FILE}: {e}")
    exit(1)

# Define forward hook
def make_hook(vec):
    v = torch.tensor(vec, dtype=torch.float32).to(DEVICE)
    def hook(module, input, output):
        # Output is a tuple: (hidden_states, (past_key_values))
        hidden_states = output[0]
        modified_hidden_states = hidden_states + ALPHA * v
        return (modified_hidden_states,) + output[1:]
    return hook

# Register hooks
print(f"Injecting AGOP 'affect' vectors into layers {LAYERS}")
hook_handles = []
for i in LAYERS:
    vec = cav_dict[i]
    handle = model.transformer.h[i].register_forward_hook(make_hook(vec))
    hook_handles.append(handle)

# Generate affect-steered sentences
print(f"Generating {NUM_SAMPLES} affect sentences without prompt...")
outputs = []
with torch.no_grad():
    for i in range(NUM_SAMPLES):
        print(f"Generating sample {i + 1}/{NUM_SAMPLES}...")
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=MAX_NEW_TOKENS,
            top_k=TOP_K,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if generated_text:
            outputs.append(generated_text)
        else:
            print(f"Warning: Sample {i + 1} produced empty text")

# Remove hooks
for handle in hook_handles:
    handle.remove()

# Print results
print("\nGenerated affect Sentences:")
print("-" * 40)
for i, text in enumerate(outputs[:NUM_SAMPLES], 1):
    print(f"{i}. {text}")
print("-" * 40)
if len(outputs) < NUM_SAMPLES:
    print(f"Warning: Only generated {len(outputs)}/{NUM_SAMPLES} samples")