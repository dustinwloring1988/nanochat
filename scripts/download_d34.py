"""
Download the pre-trained karpathy/nanochat-d34 model from HuggingFace.

This saves ~$2,500 and ~100 hours of pretraining by reusing the already-trained base model.
Run as: python -m scripts.download_d34
"""

import os
from huggingface_hub import hf_hub_download
from nanochat.common import get_base_dir

REPO_ID = "karpathy/nanochat-d34"
MODEL_STEP = 169150

def main():
    base_dir = get_base_dir()

    # Define file destinations
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", "d34")

    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Files to download: (repo_filename, local_directory)
    files = [
        ("tokenizer.pkl", tokenizer_dir),
        ("token_bytes.pt", tokenizer_dir),
        (f"model_{MODEL_STEP}.pt", checkpoint_dir),
        (f"meta_{MODEL_STEP}.json", checkpoint_dir),
    ]

    for filename, dest_dir in files:
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            print(f"Already exists, skipping: {dest_path}")
            continue
        print(f"Downloading {filename} -> {dest_path}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=dest_dir,
        )
        print(f"  Done.")

    print(f"\nPre-trained model ready at: {base_dir}")
    print(f"Run SFT fine-tuning with:")
    print(f"  torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --model-tag d34")

if __name__ == "__main__":
    main()
