# semantic_analysis_fixed.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import json
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def calculate_semantic_coherence(text, model):
    """Calculate semantic coherence metrics with fixed combination"""
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) < 2:
        return {
            'consecutive_coherence': 0.0,
            'topic_consistency': 0.0,
            'repetition_score': 0.0,
            'combined_coherence': 0.0,  # Fixed calculation
            'sentence_count': len(sentences)
        }
    
    # Get embeddings
    embeddings = model.encode(sentences)
    
    # 1. Consecutive sentence coherence
    consecutive_similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        consecutive_similarities.append(sim)
    consecutive_coherence = np.mean(consecutive_similarities)
    
    # 2. Topic consistency (similarity to first sentence)
    first_sentence = embeddings[0]
    topic_similarities = []
    for i in range(1, len(embeddings)):
        sim = cosine_similarity([first_sentence], [embeddings[i]])[0][0]
        topic_similarities.append(sim)
    topic_consistency = np.mean(topic_similarities)
    
    # 3. Repetition detection
    similarity_matrix = cosine_similarity(embeddings)
    high_sim_count = 0
    total_pairs = 0
    
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i][j] > 0.8:  # High similarity threshold
                high_sim_count += 1
            total_pairs += 1
    
    repetition_score = high_sim_count / total_pairs if total_pairs > 0 else 0
    
    # 4. FIXED: Use weighted average instead of addition
    combined_coherence = (consecutive_coherence * 0.7) + (topic_consistency * 0.3) - (repetition_score * 0.5)
    combined_coherence = max(0, combined_coherence)  # Ensure non-negative
    
    return {
        'consecutive_coherence': consecutive_coherence,
        'topic_consistency': topic_consistency,
        'repetition_score': repetition_score,
        'combined_coherence': combined_coherence,
        'sentence_count': len(sentences)
    }

def analyze_semantic_coherence_batch(json_file):
    """Analyze semantic coherence across all samples with fixed metrics"""
    
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    print(f"Processing {len(data['samples'])} samples...")
    for i, sample in enumerate(data['samples']):
        print(f"Processing sample {i+1}/{len(data['samples'])}: {sample['sample_id']}")
        
        # Debug specific sample
        if sample['sample_id'] == 'fear_a10_16':
            print(f"\nDEBUG FEAR SAMPLE FIXED CALCULATION:")
            
        try:
            metrics = calculate_semantic_coherence(sample['text'], model)
            
            # Debug output for fear sample
            if sample['sample_id'] == 'fear_a10_16':
                print(f"  Consecutive coherence: {metrics['consecutive_coherence']:.3f}")
                print(f"  Topic consistency: {metrics['topic_consistency']:.3f}")
                print(f"  Repetition score: {metrics['repetition_score']:.3f}")
                print(f"  FIXED Combined score: {metrics['combined_coherence']:.3f}")
            
            results.append({
                'sample_id': sample['sample_id'],
                'affect': sample['affect'],
                'alpha': sample['alpha'],
                'liwc_authenticity': sample['liwc_authenticity'],
                'consecutive_coherence': metrics['consecutive_coherence'],
                'topic_consistency': metrics['topic_consistency'],
                'repetition_score': metrics['repetition_score'],
                'combined_coherence': metrics['combined_coherence'],
                'sentence_count': metrics['sentence_count'],
                'text_length': len(sample['text'])
            })
            
        except Exception as e:
            print(f"Error processing {sample['sample_id']}: {e}")
            results.append({
                'sample_id': sample['sample_id'],
                'affect': sample['affect'],
                'alpha': sample['alpha'],
                'liwc_authenticity': sample['liwc_authenticity'],
                'consecutive_coherence': None,
                'topic_consistency': None,
                'repetition_score': None,
                'combined_coherence': None,
                'sentence_count': 0,
                'text_length': len(sample['text'])
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    results_df = analyze_semantic_coherence_batch('coherence_test_data.json')
    results_df.to_csv('semantic_results_fixed.csv', index=False)
    print("Fixed semantic analysis complete!")
    print("\nFixed results by affect and alpha:")
    print(results_df.groupby(['affect', 'alpha'])['combined_coherence'].mean().round(3))