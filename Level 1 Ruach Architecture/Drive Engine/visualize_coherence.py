# visualize_coherence.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_coherence_analysis():
    """Create comprehensive coherence analysis and graphs"""
    
    # Load results
    try:
        perplexity_df = pd.read_csv('perplexity_results.csv')
        semantic_df = pd.read_csv('semantic_results_fixed.csv')
    except FileNotFoundError as e:
        print(f"Missing results file: {e}")
        print("Run perplexity_analysis.py and semantic_analysis.py first!")
        return
    
    # Merge dataframes
    merged_df = pd.merge(perplexity_df, semantic_df, on=['sample_id', 'affect', 'alpha', 'liwc_authenticity'])

    # Create visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Coherence Analysis: Drive Concept Activation Vectors', fontsize=16, fontweight='bold')
    
    # 1. Perplexity vs Alpha by Affect
    ax1 = axes[0, 0]
    affects = merged_df['affect'].unique()
    for affect in affects:
        affect_data = merged_df[merged_df['affect'] == affect]
        if len(affect_data) > 1:
            grouped = affect_data.groupby('alpha')['perplexity'].mean()
            ax1.plot(grouped.index, grouped.values, marker='o', label=affect, linewidth=2)
    
    ax1.set_xlabel('Alpha Level')
    ax1.set_ylabel('Perplexity (lower = more coherent)')
    ax1.set_title('Perplexity vs Alpha Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Semantic Coherence vs Alpha by Affect
    ax2 = axes[0, 1]
    for affect in affects:
        affect_data = merged_df[merged_df['affect'] == affect]
        if len(affect_data) > 1:
            grouped = affect_data.groupby('alpha')['combined_coherence'].mean()
            ax2.plot(grouped.index, grouped.values, marker='s', label=affect, linewidth=2)
    
    ax2.set_xlabel('Alpha Level')
    ax2.set_ylabel('Semantic Coherence (higher = more coherent)')
    ax2.set_title('Semantic Coherence vs Alpha Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Authenticity vs Coherence Scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(merged_df['liwc_authenticity'], merged_df['combined_coherence'], 
                         c=merged_df['alpha'], cmap='viridis', alpha=0.7, s=60)
    ax3.set_xlabel('LIWC Authenticity (%)')
    ax3.set_ylabel('Semantic Coherence')
    ax3.set_title('Authenticity vs Coherence')
    plt.colorbar(scatter, ax=ax3, label='Alpha Level')
    ax3.grid(True, alpha=0.3)
    
    # 4. Combined Stability Index
    ax4 = axes[1, 1]
    # Create stability index (inverse perplexity + semantic coherence)
    merged_df['stability_index'] = (1 / (merged_df['perplexity'] + 1)) + merged_df['combined_coherence']
    
    for affect in affects:
        affect_data = merged_df[merged_df['affect'] == affect]
        if len(affect_data) > 1:
            grouped = affect_data.groupby('alpha')['stability_index'].mean()
            ax4.plot(grouped.index, grouped.values, marker='^', label=affect, linewidth=2)
    
    ax4.set_xlabel('Alpha Level')
    ax4.set_ylabel('Stability Index (higher = more stable)')
    ax4.set_title('Overall Stability vs Alpha Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coherence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("COHERENCE ANALYSIS SUMMARY")
    print("="*50)
    
    print("\nPerplexity by Affect and Alpha:")
    perp_summary = merged_df.groupby(['affect', 'alpha'])['perplexity'].agg(['mean', 'std']).round(2)
    print(perp_summary)
    
    print("\nSemantic Coherence by Affect and Alpha:")
    sem_summary = merged_df.groupby(['affect', 'alpha'])['combined_coherence'].agg(['mean', 'std']).round(3)
    print(sem_summary)
    
    print("\nStability Rankings (by average stability index):")
    stability_rankings = merged_df.groupby('affect')['stability_index'].mean().sort_values(ascending=False)
    print(stability_rankings.round(3))
    
    # Save comprehensive results
    merged_df.to_csv('complete_coherence_results.csv', index=False)
    print(f"\nComplete results saved to 'complete_coherence_results.csv'")

if __name__ == "__main__":
    create_coherence_analysis()
