"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer, but with Nanochat token protocol v1.0.0

This script trains a BPE tokenizer with the following structure:
- Lexical vocabulary: 0 - 31999 (32K tokens)
- Control tokens: 32000 - 32255 (256 tokens)
  * Active special tokens: 32000 - 32059
  * Reserved tokens: 32060 - 32255 (for future expansion)
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.token_protocol import (
    LEXICAL_VOCAB_SIZE,
    TOTAL_VOCAB_SIZE,
    get_all_special_tokens,
    SPECIAL_TOKENS,
    RESERVED_TOKENS,
    PROTOCOL_VERSION,
)

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer with Nanochat protocol')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--lexical-vocab-size', type=int, default=LEXICAL_VOCAB_SIZE, 
                    help=f'Lexical vocabulary size (default: {LEXICAL_VOCAB_SIZE})')
args = parser.parse_args()

print("=" * 70)
print("Nanochat Tokenizer Training")
print("=" * 70)
print(f"Protocol version: {PROTOCOL_VERSION}")
print(f"Max chars: {args.max_chars:,}")
print(f"Doc cap: {args.doc_cap:,}")
print(f"Lexical vocab size: {args.lexical_vocab_size:,}")
print(f"Control tokens: {len(SPECIAL_TOKENS)} active + {len(RESERVED_TOKENS)} reserved")
print(f"Total vocab size: {TOTAL_VOCAB_SIZE:,}")
print("=" * 70)

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer (lexical vocabulary only, without special tokens)
print("\nTraining BPE tokenizer on lexical vocabulary...")
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.lexical_vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"✓ Training time: {train_time:.2f}s")
print(f"✓ Lexical vocab size: {tokenizer.get_vocab_size()}")

# -----------------------------------------------------------------------------
# Add special tokens to the tokenizer
# Note: RustBPETokenizer.train_from_iterator already adds SPECIAL_TOKENS
# from the old tokenizer.py. We need to verify it has the right tokens.

print("\nVerifying special tokens...")
current_special = tokenizer.get_special_tokens()
expected_special = set(get_all_special_tokens().values())

# Check if we have all expected tokens
missing_tokens = expected_special - current_special
if missing_tokens:
    print(f"⚠ Warning: Missing {len(missing_tokens)} special tokens")
    print(f"  This is expected if training with new protocol for first time")
    print(f"  The tokenizer needs to be retrained with updated rustbpe")
else:
    print(f"✓ All {len(SPECIAL_TOKENS)} active special tokens present")
    print(f"✓ All {len(RESERVED_TOKENS)} reserved tokens present")

print(f"✓ Total vocabulary size: {tokenizer.get_vocab_size()}")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
print("\nSaving tokenizer...")
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# Save protocol version metadata
import json
metadata = {
    "protocol_version": PROTOCOL_VERSION,
    "lexical_vocab_size": args.lexical_vocab_size,
    "total_vocab_size": tokenizer.get_vocab_size(),
    "special_tokens": len(SPECIAL_TOKENS),
    "reserved_tokens": len(RESERVED_TOKENS),
    "training_chars": args.max_chars,
    "training_time": train_time,
}
metadata_path = os.path.join(tokenizer_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved tokenizer to {tokenizer_dir}")
print(f"✓ Saved metadata to {metadata_path}")

# -----------------------------------------------------------------------------
# Quick inline sanity check
print("\nRunning sanity checks...")
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, "Encode/decode sanity check failed!"
print("✓ Encode/decode sanity check passed")

# Test special tokens
try:
    bos_id = tokenizer.encode_special("<|bos|>")
    user_id = tokenizer.encode_special("<|user|>")
    assistant_id = tokenizer.encode_special("<|assistant|>")
    think_medium_id = tokenizer.encode_special("<|think_medium|>")
    print(f"✓ Special tokens accessible:")
    print(f"  <|bos|> = {bos_id}")
    print(f"  <|user|> = {user_id}")
    print(f"  <|assistant|> = {assistant_id}")
    print(f"  <|think_medium|> = {think_medium_id}")
except Exception as e:
    print(f"⚠ Warning: Could not encode special tokens: {e}")
    print(f"  This is expected if using old rustbpe without new tokens")

# -----------------------------------------------------------------------------
# Cache token bytes for bits-per-byte evaluation
print("\nCaching token byte counts...")
vocab_size = tokenizer.get_vocab_size()
special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0) # special tokens are not counted
    else:
        # use the raw bytes of the token: decoding to a string first corrupts
        # tokens that are not valid standalone UTF-8 (e.g. the raw bytes >= 0x80)
        num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
        token_bytes.append(num_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"✓ Saved token_bytes to {token_bytes_path}")

print("\n" + "=" * 70)
print("Tokenizer Training Complete!")
print("=" * 70)
print(f"Location: {tokenizer_dir}")
print(f"Protocol: {PROTOCOL_VERSION}")
print(f"Vocabulary: {vocab_size:,} tokens")
print("=" * 70)
