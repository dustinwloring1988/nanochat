"""
Centralized registry of all datasets used in NanoChat training.

This module provides:
- Dataset source definitions (HuggingFace repos, subsets, licenses)
- On-demand downloading with progress tracking
- License metadata for compliance
- Unified interface for both pretraining and SFT datasets
"""

import os
import json
import time
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool

from nanochat.common import get_base_dir


@dataclass
class DatasetSource:
    """Metadata for a single dataset source."""
    name: str
    repo_id: str
    license: str
    description: str
    dataset_type: str  # "pretraining" or "sft"
    subsets: List[str] = field(default_factory=lambda: ["default"])
    total_tokens_approx: Optional[int] = None  # Approximate token count if known
    
    def get_cache_dir(self, subset: str = "default") -> str:
        """Get the local cache directory for this dataset."""
        base_dir = get_base_dir()
        slug = self.repo_id.replace("/", "--")
        return os.path.join(base_dir, "curriculum_data", slug, subset)


# -----------------------------------------------------------------------------
# Pretraining Dataset Registry

PRETRAINING_SOURCES = {
    "climbmix": DatasetSource(
        name="climbmix",
        repo_id="karpathy/climbmix-400b-shuffle",
        license="CC-BY-NC-4.0 (research/development only)",
        description="400B token English pretraining corpus",
        dataset_type="pretraining",
        total_tokens_approx=400_000_000_000,
    ),
    "nemotron_v1": DatasetSource(
        name="nemotron_v1",
        repo_id="nvidia/Nemotron-Pretraining-Specialized-v1",
        license="Mixed (see dataset card)",
        description="270.7B tokens: STEM reasoning, RQA, math, scientific code",
        dataset_type="pretraining",
        subsets=[
            "Nemotron-Pretraining-RQA",
            "Nemotron-Pretraining-STEM-SFT",
            "Nemotron-Pretraining-Math-Textbooks",
            "Nemotron-Pretraining-InfiniByte-Reasoning",
            "Nemotron-Pretraining-Wiki-Rewrite",
            "Nemotron-Pretraining-Scientific-Coding"
        ],
        total_tokens_approx=270_700_000_000,
    ),
    "nemotron_v1_1": DatasetSource(
        name="nemotron_v1_1",
        repo_id="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        license="CC-BY-4.0 with additional considerations",
        description="9.3B tokens: Code concepts, algorithms, formal logic",
        dataset_type="pretraining",
        subsets=[
            "Nemotron-Pretraining-Code-Concepts",
            "Nemotron-Pretraining-Multiple-Choice",
            "Nemotron-Pretraining-Formal-Logic",
            "Nemotron-Pretraining-Unconditional-Algorithmic",
            "Nemotron-Pretraining-Economics"
        ],
        total_tokens_approx=9_300_000_000,
    ),
    "nemotron_v1_2": DatasetSource(
        name="nemotron_v1_2",
        repo_id="nvidia/Nemotron-Pretraining-Specialized-v1.2",
        license="CC-BY-4.0 and CC-BY-2.0 with DeepSeek considerations",
        description="41.8B tokens: Fact seeking, generative QA, multiple choice",
        dataset_type="pretraining",
        subsets=[
            "Nemotron-Pretraining-Fact-Seeking",
            "Nemotron-Pretraining-Multiple-Choice",
            "Nemotron-Pretraining-Generative",
            "Nemotron-Pretraining-Moral-Scenarios"
        ],
        total_tokens_approx=41_800_000_000,
    ),
}

# -----------------------------------------------------------------------------
# SFT Dataset Registry

SFT_SOURCES = {
    "smoltalk": DatasetSource(
        name="smoltalk",
        repo_id="HuggingFaceTB/smol-smoltalk",
        license="Apache-2.0",
        description="460K examples: General instruction following, conversations",
        dataset_type="sft",
    ),
    "nemotron_multilingual": DatasetSource(
        name="nemotron_multilingual",
        repo_id="nvidia/Nemotron-SFT-Multilingual-v2",
        license="CC-BY-4.0",
        description="370K examples: Hindi/Korean/Portuguese/Japanese + math/code/STEM",
        dataset_type="sft",
    ),
    "hunter_alpha": DatasetSource(
        name="hunter_alpha",
        repo_id="heegyu/Hunter-Alpha-Coding-Agent-SFT",
        license="Check dataset card",
        description="1.2K examples: Tool-using coding agent trajectories",
        dataset_type="sft",
    ),
    "nemotron_swe": DatasetSource(
        name="nemotron_swe",
        repo_id="nvidia/Nemotron-SFT-SWE-v3.5",
        license="CC-BY-4.0 with upstream licenses",
        description="5.1K examples: Repository-level software engineering agents",
        dataset_type="sft",
    ),
    "claude_fable": DatasetSource(
        name="claude_fable",
        repo_id="Bc-AI/claude-fable-5-sft-clean",
        license="Check dataset card",
        description="63 examples: Curated coding/agent examples",
        dataset_type="sft",
    ),
}

# Combined registry
ALL_SOURCES = {**PRETRAINING_SOURCES, **SFT_SOURCES}


# -----------------------------------------------------------------------------
# Download Management (adapted from dataset.py for multi-source support)

def download_parquet_shard(args: Tuple[str, str, int]) -> bool:
    """
    Download a single parquet shard.
    
    Args:
        args: Tuple of (base_url, cache_dir, shard_index)
    
    Returns:
        True if successful, False otherwise
    """
    base_url, cache_dir, shard_index = args
    
    # Construct filename and filepath
    filename = f"shard_{shard_index:05d}.parquet"
    filepath = os.path.join(cache_dir, filename)
    
    # Skip if already exists
    if os.path.exists(filepath):
        print(f"Skipping {filename} (already exists)")
        return True
    
    # Download URL
    url = f"{base_url}/{filename}"
    print(f"Downloading {filename}...")
    
    # Retry logic
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Write to temporary file first
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            
            # Move to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
            
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            
            # Clean up partial files
            for path in [temp_path, filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            
            # Exponential backoff
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    
    return False


def download_dataset_shards(
    source_name: str,
    num_shards: int = -1,
    num_workers: int = 4,
    subset: str = "default"
) -> Tuple[str, int]:
    """
    Download parquet shards for a dataset source.
    
    Args:
        source_name: Name of the dataset in the registry
        num_shards: Number of shards to download (-1 = all available)
        num_workers: Number of parallel download workers
        subset: Subset name for multi-subset datasets
    
    Returns:
        Tuple of (cache_dir, num_downloaded)
    """
    if source_name not in ALL_SOURCES:
        raise ValueError(f"Unknown dataset source: {source_name}")
    
    source = ALL_SOURCES[source_name]
    cache_dir = source.get_cache_dir(subset)
    os.makedirs(cache_dir, exist_ok=True)
    
    # For ClimbMix, we know the structure (shard_00000.parquet to shard_06542.parquet)
    if source_name == "climbmix":
        max_shard = 6542
        base_url = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
        num_shards_to_download = max_shard + 1 if num_shards == -1 else min(num_shards, max_shard + 1)
        shard_indices = list(range(num_shards_to_download))
    else:
        # For HuggingFace datasets, we'd need to query the API or use a known structure
        # For now, we'll document that other datasets should be downloaded via the HF API
        print(f"Dataset {source_name} should be downloaded via HuggingFace datasets library")
        print(f"Use: datasets.load_dataset('{source.repo_id}')")
        return cache_dir, 0
    
    # Download shards in parallel
    print(f"Downloading {len(shard_indices)} shards for {source_name} using {num_workers} workers...")
    print(f"Target directory: {cache_dir}")
    print()
    
    download_args = [(base_url, cache_dir, idx) for idx in shard_indices]
    with Pool(processes=num_workers) as pool:
        results = pool.map(download_parquet_shard, download_args)
    
    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(shard_indices)} shards to {cache_dir}")
    
    return cache_dir, successful


def get_license_info(source_names: List[str]) -> Dict[str, str]:
    """
    Get license information for a list of dataset sources.
    
    Args:
        source_names: List of dataset names
    
    Returns:
        Dictionary mapping source name to license string
    """
    return {
        name: ALL_SOURCES[name].license
        for name in source_names
        if name in ALL_SOURCES
    }


def list_parquet_files_for_source(source_name: str, subset: str = "default") -> List[str]:
    """
    List all parquet files for a given dataset source.
    
    Args:
        source_name: Name of the dataset
        subset: Subset name for multi-subset datasets
    
    Returns:
        List of full paths to parquet files
    """
    if source_name not in ALL_SOURCES:
        raise ValueError(f"Unknown dataset source: {source_name}")
    
    source = ALL_SOURCES[source_name]
    cache_dir = source.get_cache_dir(subset)
    
    if not os.path.exists(cache_dir):
        return []
    
    parquet_files = sorted([
        f for f in os.listdir(cache_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(cache_dir, f) for f in parquet_files]
    
    return parquet_paths


# -----------------------------------------------------------------------------
# Convenience functions

def get_source(name: str) -> DatasetSource:
    """Get a dataset source by name."""
    if name not in ALL_SOURCES:
        raise ValueError(f"Unknown dataset source: {name}. Available: {list(ALL_SOURCES.keys())}")
    return ALL_SOURCES[name]


def list_pretraining_sources() -> List[str]:
    """List all available pretraining dataset names."""
    return list(PRETRAINING_SOURCES.keys())


def list_sft_sources() -> List[str]:
    """List all available SFT dataset names."""
    return list(SFT_SOURCES.keys())


def print_registry_info():
    """Print information about all registered datasets."""
    print("=" * 80)
    print("PRETRAINING DATASETS")
    print("=" * 80)
    for name, source in PRETRAINING_SOURCES.items():
        tokens_str = f"{source.total_tokens_approx/1e9:.1f}B" if source.total_tokens_approx else "unknown"
        print(f"\n{name}:")
        print(f"  Repo: {source.repo_id}")
        print(f"  Tokens: ~{tokens_str}")
        print(f"  License: {source.license}")
        print(f"  Description: {source.description}")
        if len(source.subsets) > 1:
            print(f"  Subsets: {', '.join(source.subsets)}")
    
    print("\n" + "=" * 80)
    print("SFT DATASETS")
    print("=" * 80)
    for name, source in SFT_SOURCES.items():
        print(f"\n{name}:")
        print(f"  Repo: {source.repo_id}")
        print(f"  License: {source.license}")
        print(f"  Description: {source.description}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage NanoChat dataset registry")
    parser.add_argument("--list", action="store_true", help="List all registered datasets")
    parser.add_argument("--download", type=str, help="Download shards for a dataset")
    parser.add_argument("-n", "--num-shards", type=int, default=10, help="Number of shards to download")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--subset", type=str, default="default", help="Dataset subset to download")
    
    args = parser.parse_args()
    
    if args.list:
        print_registry_info()
    elif args.download:
        download_dataset_shards(args.download, args.num_shards, args.num_workers, args.subset)
    else:
        parser.print_help()
