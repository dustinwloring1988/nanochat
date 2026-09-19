"""
Verify the Nemotron dataset subsets and analyze their characteristics.
"""

from datasets import load_dataset
import numpy as np

def check_subset(dataset_name, subset_name, num_samples=100):
    """Check if a subset exists and get basic stats."""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Subset: {subset_name}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset(
            dataset_name,
            subset_name,
            split="train",
            streaming=True
        )
        
        text_lengths = []
        sample_count = 0
        
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            
            text = example.get('text', '')
            text_lengths.append(len(text))
            sample_count += 1
            
            if i < 2:  # Show first 2 samples
                print(f"\nSample {i} ({len(text)} chars):")
                print(f"  {text[:300]}{'...' if len(text) > 300 else ''}")
        
        if text_lengths:
            print(f"\n✅ Subset found! Stats from {sample_count} samples:")
            print(f"  Character lengths:")
            print(f"    Min:    {np.min(text_lengths):>10,}")
            print(f"    Median: {np.median(text_lengths):>10,.0f}")
            print(f"    Mean:   {np.mean(text_lengths):>10,.0f}")
            print(f"    P95:    {np.percentile(text_lengths, 95):>10,.0f}")
            print(f"    Max:    {np.max(text_lengths):>10,}")
            print(f"\n  Estimated token lengths (@ ~4 chars/token):")
            print(f"    Median: ~{np.median(text_lengths)/4:>10,.0f} tokens")
            print(f"    P95:    ~{np.percentile(text_lengths, 95)/4:>10,.0f} tokens")
            
            return True
        else:
            print("❌ No samples found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("Verifying Nemotron Dataset Subsets for Training Pipeline")
    print("="*70)
    
    # Stage 1: Code and Scientific Coding (2048 tokens)
    print("\n" + "="*70)
    print("STAGE 1: Code-focused pretraining (2048 token context)")
    print("="*70)
    
    stage1_datasets = [
        ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Code-Concepts"),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Scientific-Coding"),
    ]
    
    stage1_valid = []
    for dataset, subset in stage1_datasets:
        if check_subset(dataset, subset):
            stage1_valid.append((dataset, subset))
    
    # Stage 2: Long context math and reasoning (longer context)
    print("\n" + "="*70)
    print("STAGE 2: Long-context math & reasoning pretraining")
    print("="*70)
    
    stage2_datasets = [
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Math-Textbooks"),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-InfiniByte-Reasoning"),
    ]
    
    stage2_valid = []
    for dataset, subset in stage2_datasets:
        if check_subset(dataset, subset):
            stage2_valid.append((dataset, subset))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nStage 1 (Code-focused, 2048 tokens): {len(stage1_valid)}/{len(stage1_datasets)} subsets valid")
    for dataset, subset in stage1_valid:
        print(f"  ✅ {dataset} / {subset}")
    
    print(f"\nStage 2 (Long-context math, >2048 tokens): {len(stage2_valid)}/{len(stage2_datasets)} subsets valid")
    for dataset, subset in stage2_valid:
        print(f"  ✅ {dataset} / {subset}")
    
    total_valid = len(stage1_valid) + len(stage2_valid)
    total_expected = len(stage1_datasets) + len(stage2_datasets)
    
    if total_valid == total_expected:
        print(f"\n✅ All {total_valid} dataset subsets verified successfully!")
    else:
        print(f"\n⚠️  Only {total_valid}/{total_expected} subsets verified")

if __name__ == "__main__":
    main()
