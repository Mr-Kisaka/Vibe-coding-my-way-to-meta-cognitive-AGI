# generate_with_random_vectors_debug.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Configuration
MODEL_NAME = ""  # Update model name
LAYERS = list(range(5, 21))  # Same layers as affect vector experiment
ALPHA = 15  # Same steering strength
NUM_SAMPLES = 3  # Generate 3 sentences
MAX_NEW_TOKENS = 200  # Match dataset's <200-word constraint
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

# Generate random vectors with same dimensionality as model hidden states
print("Generating random vectors...")
HIDDEN_SIZE = model.config.hidden_size
print(f"Model hidden size: {HIDDEN_SIZE}")

random_vector_dict = {}
for i in LAYERS:
    # Generate random vector with same normalization as affect vectors
    vec = np.random.randn(HIDDEN_SIZE).astype(np.float32)
    # Normalize and scale to match affect vector preprocessing
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm * 2.0  # Same scaling as affect vectors
    random_vector_dict[i] = vec
    print(f"Layer {i}: Random vector norm after scaling: {np.linalg.norm(vec):.4f}")

print(f"Generated random vectors for layers: {list(random_vector_dict.keys())}")

# Define forward hook with minimal debugging
def make_hook(vec, layer_idx):
    v = torch.tensor(vec, dtype=torch.float32).to(DEVICE)
    
    hook_called = False  # Track if hook is called
    
    def hook(module, input, output):
        nonlocal hook_called
        if not hook_called:
            print(f"Hook firing for layer {layer_idx}")
            hook_called = True
            
        # Output is a tuple: (hidden_states, (past_key_values))
        hidden_states = output[0]
        modified_hidden_states = hidden_states + ALPHA * v
        
        return (modified_hidden_states,) + output[1:]
    return hook

# Register hooks
print(f"\nRegistering hooks for layers {LAYERS} with ALPHA={ALPHA}")
hook_handles = []
for i in LAYERS:
    vec = random_vector_dict[i]
    handle = model.transformer.h[i].register_forward_hook(make_hook(vec, i))
    hook_handles.append(handle)

print(f"Registered {len(hook_handles)} hooks.")

# Generate random-vector-steered sentences
print(f"\nGenerating {NUM_SAMPLES} random-vector-steered sentences without prompt...")
outputs = []

with torch.no_grad():
    for i in range(NUM_SAMPLES):
        print(f"\n=== Generating sample {i + 1}/{NUM_SAMPLES} ===")
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)
        
        print("Starting generation...")
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
        print("Generation complete.")
        
        generated_text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if generated_text:
            outputs.append(generated_text)
        else:
            print(f"Warning: Sample {i + 1} produced empty text")

# Remove hooks
print(f"\nRemoving {len(hook_handles)} hooks...")
for handle in hook_handles:
    handle.remove()
print("All hooks removed.")

# Print results
print("\nGenerated Random-Vector-Steered Sentences:")
print("=" * 50)
for i, text in enumerate(outputs[:NUM_SAMPLES], 1):
    print(f"\nSample {i}:")
    print(f"{text}")
    print("-" * 30)

print(f"\nTotal samples generated: {len(outputs)}/{NUM_SAMPLES}")

if len(outputs) < NUM_SAMPLES:
    print(f"Warning: Only generated {len(outputs)}/{NUM_SAMPLES} samples")
    
print("\nDebugging complete. Check the norm change outputs above to verify injection is working.")