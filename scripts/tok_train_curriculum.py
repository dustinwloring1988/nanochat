"""
Train tokenizer on mixed corpus from multiple dataset sources.

This script builds a representative training corpus by sampling from multiple
pretraining sources, then trains the BPE tokenizer on this mixed corpus.

Run as:
    python -m scripts.tok_train_curriculum --mix default --total-chars 2000000000

Or for fast testing with ClimbMix only:
    python -m scripts.tok_train_curriculum --mix minimal --total-chars 500000000
"""

import argparse
import os
import time

import torch

from nanochat.common import get_base_dir, print_banner
from nanochat.tokenizer import RustBPETokenizer
from nanochat.tokenizer_corpus import (
    build_mixed_tokenizer_corpus,
    get_default_pretraining_mix,
    get_minimal_test_mix,
)

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Train tokenizer on mixed curriculum corpus")
parser.add_argument(
    "--mix",
    type=str,
    default="default",
    choices=["default", "minimal"],
    help="Corpus mixing strategy: default (pretraining sources only) or minimal (ClimbMix only)"
)
parser.add_argument(
    "--total-chars",
    type=int,
    default=2_000_000_000,
    help="Target total characters for tokenizer training corpus (default: 2B)"
)
parser.add_argument(
    "--shards-per-source",
    type=int,
    default=3,
    help="Number of shards to download per source (default: 3, ~750MB per source)"
)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=32768,
    help="Tokenizer vocabulary size (default: 32768 = 2^15)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for corpus sampling"
)
args = parser.parse_args()
print_banner()

# -----------------------------------------------------------------------------
# Select mixing strategy

if args.mix == "default":
    print("Using DEFAULT multi-source curriculum mix:")
    print("  60% ClimbMix (general web text)")
    print("  16% Nemotron v1 (STEM reasoning, math, code)")
    print("  12% Nemotron v1.1 (code concepts, algorithms)")
    print("  12% Nemotron v1.2 (fact-seeking, QA)")
    print("  No SFT sources are included; SFT requires a separately approved manifest")
    print()
    source_ratios = get_default_pretraining_mix()
else:
    print("Using MINIMAL ClimbMix-only mix for fast testing")
    print()
    source_ratios = get_minimal_test_mix()

# -----------------------------------------------------------------------------
# Build mixed corpus

print("=" * 80)
print("BUILDING MIXED TOKENIZER CORPUS")
print("=" * 80)
print(f"Target corpus size: {args.total_chars:,} characters")
print(f"Downloading {args.shards_per_source} shards per source")
print()

documents, corpus_path = build_mixed_tokenizer_corpus(
    source_ratios=source_ratios,
    total_chars=args.total_chars,
    download_shards_per_source=args.shards_per_source,
    seed=args.seed,
)

print()
print("=" * 80)
print("TRAINING TOKENIZER")
print("=" * 80)
print(f"Corpus file: {corpus_path}")
print(f"Vocabulary size: {args.vocab_size:,}")
print()

# -----------------------------------------------------------------------------
# Train tokenizer (using RustBPE tokenizer)

print("Training tokenizer with RustBPE...")
print(f"This will take a few minutes for {args.total_chars/1e9:.1f}B characters...")
print()

# Create text iterator from corpus file
def text_iterator():
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

# Train tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")
print()

# Save tokenizer
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)
print(f"Tokenizer saved to: {tokenizer_dir}")

# Save token_bytes mapping for efficient evaluation
vocab_size = tokenizer.get_vocab_size()
special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0)  # special tokens are not counted
    else:
        num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
        token_bytes.append(num_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to: {token_bytes_path}")

print()
print("=" * 80)
print("TOKENIZER TRAINING COMPLETE")
print("=" * 80)
print(f"Tokenizer saved to: {tokenizer_dir}")
print(f"Vocabulary size: {args.vocab_size:,}")
print()
print("The tokenizer is now ready for curriculum-based pretraining!")
print()
print("Next steps:")
print("  1. Verify tokenizer: python -m scripts.tok_eval")
print("  2. Start pretraining: bash runs/curriculum_4060ti.sh")
print("=" * 80)
