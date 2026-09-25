"""
Mixed corpus builder for tokenizer training.

This module creates a representative training corpus for the tokenizer by sampling
from multiple dataset sources according to specified mixing ratios. This ensures
the tokenizer vocabulary effectively represents all types of content the model will see.
"""

import os
import random
from typing import Dict, List, Tuple, Optional
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.data_registry import (
    get_source,
    list_parquet_files_for_source,
    download_dataset_shards,
)


def _download_huggingface_dataset(source, num_samples_per_subset: int = 10000):
    """
    Download and convert HuggingFace dataset to parquet format.
    
    For test/development, we download a limited sample from each subset
    rather than the full multi-TB datasets.
    
    Args:
        source: DatasetSource object
        num_samples_per_subset: Number of examples to download per subset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: HuggingFace datasets library not installed!")
        print("Install with: pip install datasets")
        return
    
    print(f"Loading samples from {source.repo_id}...")
    print(f"Dataset type: {source.dataset_type}")
    print(f"Downloading {num_samples_per_subset} examples per subset")
    
    # Download and convert each subset
    for subset in source.subsets:
        cache_dir = source.get_cache_dir(subset)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if we already have parquet files
        existing_files = list_parquet_files_for_source(source.name, subset)
        if len(existing_files) > 0:
            print(f"  {subset}: Already have {len(existing_files)} parquet files, skipping")
            continue
        
        print(f"  Downloading subset: {subset} ({num_samples_per_subset} examples)...")
        
        try:
            # Load dataset from HuggingFace
            # For single-subset datasets, don't pass the 'name' parameter
            if len(source.subsets) == 1 and subset == "default":
                dataset = load_dataset(
                    source.repo_id,
                    split='train',
                    streaming=True
                )
            else:
                dataset = load_dataset(
                    source.repo_id,
                    name=subset,
                    split='train',
                    streaming=True
                )
            
            documents = []
            for i, example in enumerate(dataset):
                if i >= num_samples_per_subset:
                    break
                
                # Extract text content based on dataset type
                text_content = None
                
                if source.dataset_type == "pretraining":
                    # Pretraining datasets: simple text field
                    if 'text' in example:
                        text_content = example['text']
                    elif 'content' in example:
                        text_content = example['content']
                
                elif source.dataset_type == "sft":
                    # SFT datasets: conversation format or instruction format
                    if 'messages' in example:
                        # Conversation format: join messages
                        messages = example['messages']
                        if isinstance(messages, list):
                            parts = []
                            for msg in messages:
                                if isinstance(msg, dict) and 'content' in msg:
                                    role = msg.get('role', 'unknown')
                                    parts.append(f"{role}: {msg['content']}")
                            text_content = "\n".join(parts)
                    elif 'conversations' in example:
                        # Alternative conversation format
                        convs = example['conversations']
                        if isinstance(convs, list):
                            parts = []
                            for conv in convs:
                                if isinstance(conv, dict) and 'value' in conv:
                                    role = conv.get('from', 'unknown')
                                    parts.append(f"{role}: {conv['value']}")
                            text_content = "\n".join(parts)
                    elif 'prompt' in example and 'response' in example:
                        # Instruction format
                        text_content = f"Instruction: {example['prompt']}\nResponse: {example['response']}"
                    elif 'text' in example:
                        # Fallback to text field
                        text_content = example['text']
                
                if text_content:
                    documents.append(text_content)
                
                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"    Loaded {i + 1}/{num_samples_per_subset} examples...")
            
            # Save as parquet shards (split into multiple files for compatibility with training loop)
            if documents:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                # Split into 3-5 shards for better distribution during training
                num_shards = min(5, max(1, len(documents) // 2000))
                chunk_size = len(documents) // num_shards + 1
                
                for shard_idx in range(num_shards):
                    start_idx = shard_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(documents))
                    
                    if start_idx >= len(documents):
                        break
                    
                    shard_docs = documents[start_idx:end_idx]
                    shard_filename = f"shard_{shard_idx:05d}.parquet"
                    shard_path = os.path.join(cache_dir, shard_filename)
                    
                    # Create PyArrow table and save
                    table = pa.table({'text': shard_docs})
                    pq.write_table(table, shard_path)
                    
                    print(f"    Saved {len(shard_docs)} docs to {shard_filename}")
                
                print(f"  ✓ Completed {subset}: {len(documents)} documents in {num_shards} shards")
            else:
                print(f"  ⚠ No documents found for {subset}")
        
        except Exception as e:
            print(f"  ✗ Error downloading {subset}: {e}")
            print("    Continuing with other subsets...")
            continue


def sample_documents_from_source(
    source_name: str,
    target_chars: int,
    subset: str = "default",
    seed: int = 42
) -> List[str]:
    """
    Sample documents from a data source to reach approximately target_chars.
    
    Args:
        source_name: Name of the dataset source
        target_chars: Target number of characters to sample
        subset: Subset name for multi-subset datasets
        seed: Random seed for reproducible sampling
    
    Returns:
        List of document strings
    """
    random.seed(seed)
    
    # Get parquet files for this source
    parquet_files = list_parquet_files_for_source(source_name, subset)
    
    if not parquet_files:
        print(f"Warning: No parquet files found for {source_name}/{subset}")
        return []
    
    documents = []
    total_chars = 0
    
    # Shuffle files for random sampling
    random.shuffle(parquet_files)
    
    for filepath in parquet_files:
        if total_chars >= target_chars:
            break
        
        try:
            pf = pq.ParquetFile(filepath)
            
            # Sample row groups randomly
            row_group_indices = list(range(pf.num_row_groups))
            random.shuffle(row_group_indices)
            
            for rg_idx in row_group_indices:
                if total_chars >= target_chars:
                    break
                
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                
                for text in texts:
                    documents.append(text)
                    total_chars += len(text)
                    
                    if total_chars >= target_chars:
                        break
        
        except Exception as e:
            print(f"Warning: Error reading {filepath}: {e}")
            continue
    
    print(f"Sampled {len(documents):,} documents ({total_chars:,} chars) from {source_name}/{subset}")
    
    return documents


def build_mixed_tokenizer_corpus(
    source_ratios: Dict[str, float],
    total_chars: int = 2_000_000_000,  # 2B characters default
    output_path: Optional[str] = None,
    download_shards_per_source: int = 3,
    seed: int = 42
) -> Tuple[List[str], str]:
    """
    Build a mixed tokenizer training corpus from multiple sources.
    
    Args:
        source_ratios: Dictionary mapping source name to mixing ratio
        total_chars: Target total number of characters
        output_path: Path to save the corpus (optional)
        download_shards_per_source: Number of shards to download per source
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (documents, corpus_path)
    """
    # Validate ratios
    total_ratio = sum(source_ratios.values())
    assert 0.99 <= total_ratio <= 1.01, f"Source ratios must sum to ~1.0, got {total_ratio}"
    
    # Download minimal shards for each source
    print("=" * 80)
    print("DOWNLOADING TOKENIZER CORPUS DATA")
    print("=" * 80)
    for source_name in source_ratios.keys():
        source = get_source(source_name)
        print(f"\nDownloading data for {source_name}...")
        
        if source_name == "climbmix":
            # ClimbMix uses the special download function
            print(f"Downloading {download_shards_per_source} shards from ClimbMix...")
            download_dataset_shards(
                source_name,
                num_shards=download_shards_per_source,
                num_workers=4
            )
        else:
            # For HuggingFace datasets (Nemotron pretraining + SFT), download samples
            # Adjust sample size based on source importance and type
            if source_name == "nemotron_v1":
                samples_per_subset = 10000  # More for main STEM source
            elif source_name.startswith("nemotron_v1"):
                samples_per_subset = 5000   # Less for supplementary Nemotron sources
            elif source.dataset_type == "sft":
                # SFT datasets: smaller samples since they're more focused
                samples_per_subset = min(2000, 5000)  # Cap at 2K for small datasets
            else:
                samples_per_subset = 5000
            
            print(f"Downloading {samples_per_subset} examples from {source.repo_id}")
            _download_huggingface_dataset(source, samples_per_subset)
    
    print("\n" + "=" * 80)
    print("SAMPLING DOCUMENTS FROM EACH SOURCE")
    print("=" * 80)
    
    # Sample documents from each source
    all_documents = []
    
    for source_name, ratio in source_ratios.items():
        target_chars_for_source = int(total_chars * ratio)
        print(f"\nSampling {target_chars_for_source:,} chars from {source_name}...")
        
        # For sources with subsets, we might want to sample from each
        source = get_source(source_name)
        
        if len(source.subsets) == 1 and source.subsets[0] == "default":
            # Single subset source
            docs = sample_documents_from_source(
                source_name,
                target_chars_for_source,
                "default",
                seed
            )
            all_documents.extend(docs)
        else:
            # Multi-subset source - distribute evenly across subsets
            chars_per_subset = target_chars_for_source // len(source.subsets)
            for subset in source.subsets:
                docs = sample_documents_from_source(
                    source_name,
                    chars_per_subset,
                    subset,
                    seed
                )
                all_documents.extend(docs)
    
    # Shuffle all documents for good mixing
    random.seed(seed)
    random.shuffle(all_documents)
    
    print(f"\nTotal documents in mixed corpus: {len(all_documents):,}")
    total_chars_actual = sum(len(doc) for doc in all_documents)
    print(f"Total characters: {total_chars_actual:,}")
    
    # Save to file if output_path provided
    if output_path is None:
        base_dir = get_base_dir()
        output_path = os.path.join(base_dir, "tokenizer_corpus_mixed.txt")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nSaving corpus to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in all_documents:
            # Write each document on its own line (or separated by double newline)
            f.write(doc)
            f.write('\n\n')
    
    print("Corpus saved successfully!")
    
    return all_documents, output_path


def get_default_pretraining_mix() -> Dict[str, float]:
    """
    Get the default mixing ratios for the tokenizer corpus.

    Returns a pretraining-only mix; SFT sources require a separate approved
    manifest and are never downloaded by this default path.
    """
    return {
        "climbmix": 0.60,
        "nemotron_v1": 0.16,
        "nemotron_v1_1": 0.12,
        "nemotron_v1_2": 0.12,
    }


def get_minimal_test_mix() -> Dict[str, float]:
    """
    Get a minimal mixing ratio for fast testing (ClimbMix only).
    
    Returns:
        Dictionary with just ClimbMix
    """
    return {"climbmix": 1.0}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build mixed tokenizer training corpus")
    parser.add_argument(
        "--mix",
        type=str,
        default="default",
        choices=["default", "minimal"],
        help="Mixing strategy: default (multi-source) or minimal (ClimbMix only)"
    )
    parser.add_argument(
        "--total-chars",
        type=int,
        default=2_000_000_000,
        help="Target total characters (default: 2B)"
    )
    parser.add_argument(
        "--shards-per-source",
        type=int,
        default=3,
        help="Number of shards to download per source (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for corpus file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Select mixing strategy
    if args.mix == "default":
        source_ratios = get_default_pretraining_mix()
        print("Using default multi-source mix for tokenizer corpus")
    else:
        source_ratios = get_minimal_test_mix()
        print("Using minimal ClimbMix-only mix for tokenizer corpus")
    
    # Build corpus
    documents, corpus_path = build_mixed_tokenizer_corpus(
        source_ratios=source_ratios,
        total_chars=args.total_chars,
        output_path=args.output,
        download_shards_per_source=args.shards_per_source,
        seed=args.seed
    )
    
    print("\n" + "=" * 80)
    print("CORPUS BUILD COMPLETE")
    print("=" * 80)
    print(f"Corpus saved to: {corpus_path}")
    print(f"Total documents: {len(documents):,}")
    print("Ready for tokenizer training!")
