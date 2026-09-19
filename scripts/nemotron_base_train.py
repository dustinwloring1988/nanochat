"""
Base pretraining using Nemotron datasets.
This is a modified version of base_train.py that uses Nemotron data instead of ClimbMix.

Run as:
torchrun --standalone --nproc_per_node=1 -m scripts.nemotron_base_train -- --depth=8 --num-iterations=200
"""

import sys
import os

# Inject Nemotron dataloader into the dataset module
print("Configuring Nemotron datasets...")

# Get stage from environment variable (set by the pipeline script)
stage = os.environ.get("NEMOTRON_STAGE", "stage1")
print(f"Using Nemotron {stage} datasets")

# Remove --stage argument from sys.argv if present (it's not recognized by base_train)
new_argv = []
skip_next = False
for i, arg in enumerate(sys.argv):
    if skip_next:
        skip_next = False
        continue
    if arg.startswith("--stage"):
        if "=" in arg:
            # --stage=value format
            _, value = arg.split("=", 1)
            stage = value
        else:
            # --stage value format
            skip_next = True
            if i + 1 < len(sys.argv):
                stage = sys.argv[i + 1]
        continue
    new_argv.append(arg)

sys.argv = new_argv

# Monkey-patch the dataset module to use Nemotron data
import nanochat.dataset as dataset_module
from nanochat.nemotron_dataloader import nemotron_iter_batched

# Replace the parquets_iter_batched function
original_parquets_iter_batched = dataset_module.parquets_iter_batched

def parquets_iter_batched_wrapper(split, start=0, step=1):
    """Wrapper that injects the stage parameter"""
    return nemotron_iter_batched(split=split, stage=stage, start=start, step=step)

dataset_module.parquets_iter_batched = parquets_iter_batched_wrapper

# Also patch in dataloader module
import nanochat.dataloader as dataloader_module
dataloader_module.parquets_iter_batched = parquets_iter_batched_wrapper

print("✓ Nemotron datasets configured")
print()

# Now import and run the standard base_train script
from scripts import base_train
