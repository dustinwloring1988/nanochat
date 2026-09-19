"""
Dataloader for Nemotron datasets that's compatible with nanochat's training pipeline.
This replaces the ClimbMix dataset with Nemotron code/math/reasoning datasets.
"""

import torch
from datasets import load_dataset, interleave_datasets
from nanochat.tokenizer import get_tokenizer


class NemotronDataGenerator:
    """
    Generator that yields tokenized text from Nemotron datasets.
    Compatible with nanochat's dataloader interface.
    """
    
    def __init__(self, split="train", stage="stage1"):
        """
        Args:
            split: "train" or "val" (we'll use train for now, val TBD)
            stage: "stage1" (code), "stage2a" (math), or "stage2b" (reasoning)
        """
        self.split = split
        self.stage = stage
        self.tokenizer = get_tokenizer()
        
        # Define dataset configurations for each stage
        if stage == "stage1":
            # Stage 1: Code-focused (2048 token context)
            dataset_configs = [
                {
                    "name": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
                    "subset": "Nemotron-Pretraining-Code-Concepts",
                    "weight": 0.5,
                },
                {
                    "name": "nvidia/Nemotron-Pretraining-Specialized-v1",
                    "subset": "Nemotron-Pretraining-Scientific-Coding",
                    "weight": 0.5,
                },
            ]
        elif stage == "stage2a":
            # Stage 2a: Math textbooks (8192 token context)
            dataset_configs = [
                {
                    "name": "nvidia/Nemotron-Pretraining-Specialized-v1",
                    "subset": "Nemotron-Pretraining-Math-Textbooks",
                    "weight": 1.0,
                },
            ]
        elif stage == "stage2b":
            # Stage 2b: Long context reasoning (32768 tokens)
            dataset_configs = [
                {
                    "name": "nvidia/Nemotron-Pretraining-Specialized-v1",
                    "subset": "Nemotron-Pretraining-InfiniByte-Reasoning",
                    "weight": 1.0,
                },
            ]
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        print(f"\n{'='*70}")
        print(f"Loading Nemotron {stage} datasets...")
        print(f"{'='*70}")
        
        # Load and interleave datasets
        datasets = []
        weights = []
        
        for config in dataset_configs:
            dataset_name = config["name"]
            subset_name = config["subset"]
            weight = config["weight"]
            
            print(f"\n  Loading: {subset_name}")
            print(f"    Weight: {weight}")
            
            try:
                ds = load_dataset(
                    dataset_name,
                    subset_name,
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )
                datasets.append(ds)
                weights.append(weight)
                print(f"    ✓ Loaded successfully")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                raise
        
        # Interleave datasets with specified weights
        if len(datasets) > 1:
            print(f"\n  Interleaving {len(datasets)} datasets...")
            self.dataset = interleave_datasets(
                datasets,
                probabilities=weights,
                seed=42,
                stopping_strategy="all_exhausted"
            )
        else:
            self.dataset = datasets[0]
        
        print(f"\n{'='*70}")
        print(f"Nemotron {stage} dataset ready!")
        print(f"{'='*70}\n")
    
    def __iter__(self):
        """
        Iterate over documents, yielding raw text strings.
        This matches the interface expected by nanochat's dataloader.
        """
        for example in self.dataset:
            # Extract text from the example
            text = example.get("text", "")
            
            # Skip empty documents
            if not text or len(text.strip()) < 10:
                continue
            
            yield text


def nemotron_iter_batched(split="train", stage="stage1", start=0, step=1):
    """
    Drop-in replacement for parquets_iter_batched that uses Nemotron datasets.
    Yields batches of text documents.
    
    Args:
        split: "train" or "val"
        stage: "stage1", "stage2a", or "stage2b"
        start: starting index for DDP (usually rank)
        step: step size for DDP (usually world_size)
    
    Yields:
        List of text strings (documents)
    """
    generator = NemotronDataGenerator(split=split, stage=stage)
    
    batch = []
    batch_size = 100  # Batch documents for efficiency
    
    for idx, text in enumerate(generator):
        # DDP sharding: only yield documents for this rank
        if idx % step != start:
            continue
        
        batch.append(text)
        
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    # Yield remaining documents
    if batch:
        yield batch


# For compatibility with the standard training script
def parquets_iter_batched_nemotron(split, start=0, step=1, stage="stage1"):
    """
    Wrapper that matches the signature of the standard parquets_iter_batched.
    """
    return nemotron_iter_batched(split=split, stage=stage, start=start, step=step)
