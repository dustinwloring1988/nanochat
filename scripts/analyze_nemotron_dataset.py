"""
Analyze Nvidia Nemotron-Pretraining-Specialized-v1.2 dataset to determine
appropriate context length for pretraining.
"""

from datasets import load_dataset
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_subset(subset_name, num_samples=1000):
    """Analyze a single subset of the dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing subset: {subset_name}")
    print(f"{'='*60}")
    
    try:
        # Load the dataset subset
        dataset = load_dataset(
            "nvidia/Nemotron-Pretraining-Specialized-v1.2",
            subset_name,
            split="train",
            streaming=True
        )
        
        text_lengths = []
        samples = []
        
        # Sample from the dataset
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            
            text = example.get('text', '')
            text_len = len(text)
            text_lengths.append(text_len)
            
            # Store a few samples for inspection
            if i < 5:
                samples.append({
                    'index': i,
                    'char_length': text_len,
                    'preview': text[:200] + "..." if len(text) > 200 else text
                })
        
        if not text_lengths:
            print(f"No data found in subset {subset_name}")
            return None
        
        # Calculate statistics
        stats = {
            'subset': subset_name,
            'num_samples': len(text_lengths),
            'min_length': np.min(text_lengths),
            'max_length': np.max(text_lengths),
            'mean_length': np.mean(text_lengths),
            'median_length': np.median(text_lengths),
            'p25': np.percentile(text_lengths, 25),
            'p75': np.percentile(text_lengths, 75),
            'p90': np.percentile(text_lengths, 90),
            'p95': np.percentile(text_lengths, 95),
            'p99': np.percentile(text_lengths, 99),
        }
        
        # Print statistics
        print(f"\nStatistics (character lengths):")
        print(f"  Samples analyzed: {stats['num_samples']}")
        print(f"  Min length:       {stats['min_length']:>10,.0f}")
        print(f"  P25:              {stats['p25']:>10,.0f}")
        print(f"  Median:           {stats['median_length']:>10,.0f}")
        print(f"  Mean:             {stats['mean_length']:>10,.0f}")
        print(f"  P75:              {stats['p75']:>10,.0f}")
        print(f"  P90:              {stats['p90']:>10,.0f}")
        print(f"  P95:              {stats['p95']:>10,.0f}")
        print(f"  P99:              {stats['p99']:>10,.0f}")
        print(f"  Max length:       {stats['max_length']:>10,.0f}")
        
        print(f"\nSample previews:")
        for sample in samples:
            print(f"\n  Sample {sample['index']} ({sample['char_length']:,} chars):")
            print(f"    {sample['preview']}")
        
        return stats, text_lengths
        
    except Exception as e:
        print(f"Error loading subset {subset_name}: {e}")
        return None

def estimate_token_lengths(char_lengths, chars_per_token=4):
    """Estimate token lengths from character lengths."""
    # Rough estimate: ~4 chars per token for English text
    token_lengths = [length / chars_per_token for length in char_lengths]
    return token_lengths

def recommend_context_length(all_stats):
    """Recommend appropriate context length based on analysis."""
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Collect all median lengths
    medians = [s['median_length'] for s in all_stats]
    p95s = [s['p95'] for s in all_stats]
    
    overall_median_chars = np.median(medians)
    overall_p95_chars = np.median(p95s)
    
    # Estimate tokens (rough: 4 chars per token)
    median_tokens = overall_median_chars / 4
    p95_tokens = overall_p95_chars / 4
    
    print(f"\nOverall statistics across subsets:")
    print(f"  Median text length: ~{overall_median_chars:,.0f} chars (~{median_tokens:,.0f} tokens)")
    print(f"  P95 text length:    ~{overall_p95_chars:,.0f} chars (~{p95_tokens:,.0f} tokens)")
    
    print(f"\nContext length recommendations:")
    
    # Common power-of-2 context lengths
    context_options = [512, 1024, 2048, 4096, 8192]
    
    for ctx_len in context_options:
        coverage_median = min(100, (ctx_len / median_tokens) * 100)
        coverage_p95 = min(100, (ctx_len / p95_tokens) * 100)
        
        if ctx_len <= median_tokens:
            status = "❌ Too small - will truncate most documents"
        elif ctx_len < p95_tokens:
            status = "⚠️  Adequate - will truncate longer documents"
        elif ctx_len >= p95_tokens * 1.2:
            status = "✅ Good - captures 95%+ of documents fully"
        else:
            status = "✅ Recommended - balances coverage and efficiency"
        
        print(f"  {ctx_len:>5} tokens: {status}")
        print(f"         (~{coverage_median:.0f}% of median, ~{coverage_p95:.0f}% of P95)")

def main():
    print("Analyzing Nvidia Nemotron-Pretraining-Specialized-v1.2 Dataset")
    print("This may take a few minutes...\n")
    
    # The dataset has 4 subsets
    subsets = [
        "Nemotron-Pretraining-Fact-Seeking",
        "Nemotron-Pretraining-Moral-Scenarios",
        "Nemotron-Pretraining-Generative",
        "Nemotron-Pretraining-Multiple-Choice"
    ]
    
    all_stats = []
    all_lengths = defaultdict(list)
    
    for subset in subsets:
        result = analyze_subset(subset, num_samples=1000)
        if result:
            stats, lengths = result
            all_stats.append(stats)
            all_lengths[subset] = lengths
    
    if all_stats:
        recommend_context_length(all_stats)
        
        # Save detailed results
        print(f"\n{'='*60}")
        print("Saving detailed analysis...")
        
        with open('nemotron_dataset_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("Nemotron Dataset Analysis Results\n")
            f.write("="*60 + "\n\n")
            
            for stats in all_stats:
                f.write(f"Subset: {stats['subset']}\n")
                f.write(f"  Samples: {stats['num_samples']}\n")
                f.write(f"  Character lengths:\n")
                f.write(f"    Min:    {stats['min_length']:>10,.0f}\n")
                f.write(f"    P25:    {stats['p25']:>10,.0f}\n")
                f.write(f"    Median: {stats['median_length']:>10,.0f}\n")
                f.write(f"    Mean:   {stats['mean_length']:>10,.0f}\n")
                f.write(f"    P75:    {stats['p75']:>10,.0f}\n")
                f.write(f"    P90:    {stats['p90']:>10,.0f}\n")
                f.write(f"    P95:    {stats['p95']:>10,.0f}\n")
                f.write(f"    P99:    {stats['p99']:>10,.0f}\n")
                f.write(f"    Max:    {stats['max_length']:>10,.0f}\n")
                
                # Estimate tokens
                f.write(f"  Estimated token lengths (@ ~4 chars/token):\n")
                f.write(f"    Median: ~{stats['median_length']/4:>10,.0f} tokens\n")
                f.write(f"    P95:    ~{stats['p95']/4:>10,.0f} tokens\n")
                f.write("\n")
        
        print("Results saved to: nemotron_dataset_analysis.txt")

if __name__ == "__main__":
    main()
