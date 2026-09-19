"""
Multi-stage Nemotron pretraining script with Docker support.
Supports training on multiple dataset subsets with different context lengths.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader
import importlib.util

# Add nanochat to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


class NemotronDataset:
    """Dataset loader for Nemotron pretraining subsets."""
    
    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]],
        tokenizer,
        context_length: int,
        seed: int = 42,
        streaming: bool = True
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.seed = seed
        
        print(f"\n{'='*70}")
        print("Loading Nemotron datasets...")
        print(f"{'='*70}")
        
        datasets = []
        weights = []
        
        for config in dataset_configs:
            dataset_name = config["name"]
            subset_name = config["subset"]
            weight = config.get("weight", 1.0)
            
            print(f"\n  Loading: {dataset_name}")
            print(f"    Subset: {subset_name}")
            print(f"    Weight: {weight}")
            
            try:
                ds = load_dataset(
                    dataset_name,
                    subset_name,
                    split="train",
                    streaming=streaming
                )
                datasets.append(ds)
                weights.append(weight)
                print(f"    [OK] Loaded successfully")
            except Exception as e:
                print(f"    [ERROR] Error: {e}")
                raise
        
        # Interleave datasets with specified weights
        if len(datasets) > 1:
            print(f"\n  Interleaving {len(datasets)} datasets with weights {weights}")
            self.dataset = interleave_datasets(
                datasets,
                probabilities=weights,
                seed=seed,
                stopping_strategy="all_exhausted"
            )
        else:
            self.dataset = datasets[0]
        
        print(f"\n{'='*70}")
        print(f"Dataset ready! Sequence length: {context_length} tokens")
        print(f"{'='*70}\n")
    
    def __iter__(self):
        """Iterate over tokenized examples."""
        for example in self.dataset:
            text = example.get("text", "")
            
            # Tokenize
            tokens = self.tokenizer.encode(text, prepend="bos")
            
            # Truncate or skip based on length
            if len(tokens) > self.context_length:
                tokens = tokens[:self.context_length]
            
            # Skip very short sequences
            if len(tokens) < 10:
                continue
            
            # Pad if necessary (for batching)
            if len(tokens) < self.context_length:
                # Use 0 as pad token (or whatever makes sense for your tokenizer)
                tokens = tokens + [0] * (self.context_length - len(tokens))
            
            yield torch.tensor(tokens, dtype=torch.long)


def load_config(config_path: str, stage: str = None) -> Dict[str, Any]:
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Handle stage-specific config selection
    if stage == "stage2a" and hasattr(config_module, 'CONFIG_STAGE2A'):
        return config_module.CONFIG_STAGE2A
    elif stage == "stage2b" and hasattr(config_module, 'CONFIG_STAGE2B'):
        return config_module.CONFIG_STAGE2B
    elif hasattr(config_module, 'CONFIG'):
        return config_module.CONFIG
    elif hasattr(config_module, 'CONFIG_STAGE2A'):
        return config_module.CONFIG_STAGE2A
    elif hasattr(config_module, 'CONFIG_STAGE2B'):
        return config_module.CONFIG_STAGE2B
    else:
        raise ValueError("Config file must contain CONFIG, CONFIG_STAGE2A, or CONFIG_STAGE2B")


def train(config: Dict[str, Any], args: argparse.Namespace):
    """Main training function."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    log_dir = Path(config["output"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()
    
    # Update vocab size if tokenizer is different
    actual_vocab_size = tokenizer.get_vocab_size()
    config["model"]["vocab_size"] = actual_vocab_size
    print(f"Tokenizer vocab size: {actual_vocab_size}")
    
    # Create model
    print("\nInitializing model...")
    model_config = GPTConfig(**config["model"])
    model = GPT(model_config)
    
    # Load checkpoint if specified
    if "load_from" in config["output"] and Path(config["output"]["load_from"]).exists():
        print(f"Loading checkpoint from: {config['output']['load_from']}")
        checkpoint = torch.load(config["output"]["load_from"], map_location=device)
        model.load_state_dict(checkpoint["model"])
    
    model = model.to(device)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Create dataset
    dataset = NemotronDataset(
        dataset_configs=config["datasets"],
        tokenizer=tokenizer,
        context_length=config["model"]["sequence_len"],
        seed=config["data"]["seed"],
        streaming=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=0,  # Set to 0 for streaming datasets
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["optimizer"]["betas"],
        eps=config["optimizer"]["eps"]
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    model.train()
    step = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        if step >= config["training"]["max_steps"]:
            break
        
        batch = batch.to(device)
        
        # Forward pass
        logits, loss = model(batch[:, :-1], batch[:, 1:])
        loss = loss / config["training"]["gradient_accumulation_steps"]
        
        # Backward pass
        loss.backward()
        
        total_loss += loss.item()
        
        # Update weights
        if (batch_idx + 1) % config["training"]["gradient_accumulation_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            
            # Logging
            if step % 10 == 0:
                avg_loss = total_loss / 10
                print(f"Step {step}/{config['training']['max_steps']} | Loss: {avg_loss:.4f}")
                total_loss = 0.0
            
            # Evaluation
            if step % config["training"]["eval_interval"] == 0:
                print(f"\n[Step {step}] Running evaluation...")
                # TODO: Add evaluation logic
            
            # Save checkpoint
            if step % config["training"]["save_interval"] == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"
                print(f"\n[Step {step}] Saving checkpoint to {checkpoint_path}")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }, checkpoint_path)
    
    # Save final checkpoint
    final_path = checkpoint_dir / "final.pt"
    print(f"\nSaving final checkpoint to {final_path}")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }, final_path)
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Nemotron pretraining")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/nemotron_stage1_config.py)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["stage1", "stage2a", "stage2b"],
        help="Training stage (overrides config selection)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config, args.stage)
    
    print("="*70)
    print(f"Nemotron Pretraining - {config['output']['run_name']}")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Sequence length: {config['model']['sequence_len']}")
    print(f"  Datasets: {len(config['datasets'])}")
    for ds_config in config["datasets"]:
        print(f"    - {ds_config['subset']} (weight: {ds_config.get('weight', 1.0)})")
    print(f"  Max steps: {config['training']['max_steps']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    effective_batch = config['training']['batch_size'] * config['training']['gradient_accumulation_steps']
    print(f"  Effective batch size: {effective_batch}")
    
    # Train
    train(config, args)


if __name__ == "__main__":
    main()
