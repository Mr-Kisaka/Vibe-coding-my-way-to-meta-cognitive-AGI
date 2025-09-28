# perplexity_analysis.py
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import json
import numpy as np
import pandas as pd
from collections import Counter

def calculate_perplexity(text, model, tokenizer, device='cpu'):
    """Calculate perplexity score for given text"""
    
    # Tokenize
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    # Convert loss to perplexity
    perplexity = torch.exp(loss).item()
    return perplexity

def analyze_perplexity_batch(json_file):
    """Analyze perplexity across all samples"""
    
    print("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples in JSON: {len(data['samples'])}")
    print("Sample distribution:")
    expected_counts = Counter([f"{s['affect']}_a{s['alpha']}" for s in data['samples']])
    for key, count in sorted(expected_counts.items()):
        print(f"  {key}: {count} samples")
    
    results = []
    successful_count = 0
    failed_count = 0

    print(f"\nProcessing {len(data['samples'])} samples...")
    for i, sample in enumerate(data['samples']):
        print(f"Processing sample {i+1}/{len(data['samples'])}: {sample['sample_id']}")
        
        try:
            perplexity = calculate_perplexity(sample['text'], model, tokenizer, device)
            successful_count += 1
            
            results.append({
                'sample_id': sample['sample_id'],
                'affect': sample['affect'],
                'alpha': sample['alpha'],
                'liwc_authenticity': sample['liwc_authenticity'],
                'perplexity': perplexity,
                'text_length': len(sample['text'])
            })
            
        except Exception as e:
            print(f"ERROR processing {sample['sample_id']}: {e}")
            failed_count += 1
            results.append({
                'sample_id': sample['sample_id'],
                'affect': sample['affect'],
                'alpha': sample['alpha'],
                'liwc_authenticity': sample['liwc_authenticity'],
                'perplexity': None,
                'text_length': len(sample['text'])
            })
    
    print("\nProcessing Summary:")
    print(f"  Successful: {successful_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {len(data['samples'])}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    results_df = analyze_perplexity_batch('coherence_test_data.json')
    results_df.to_csv('perplexity_results.csv', index=False)
    print("\nPerplexity analysis complete!")
    print(results_df.groupby(['affect', 'alpha'])['perplexity'].mean())
