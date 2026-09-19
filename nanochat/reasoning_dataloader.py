"""
Reasoning-capable dataloader for Nemotron datasets.

This module provides dataloaders for training reasoning models using:
1. nvidia/Nemotron-Post-Training-Dataset-v2 (SFT data with reasoning_on/off)
2. nvidia/Nemotron-Cascade-SFT-Stage-2 (instruction following + reasoning)

The dataloader handles:
- Loading datasets from HuggingFace with proper splits
- Converting Nemotron format to Nanochat message format
- Mixing reasoning_on and reasoning_off examples
- Integration with existing chat tokenizer
"""

from typing import Iterator, List, Dict, Any, Optional, Tuple
from datasets import load_dataset, Dataset
import random
from nanochat.messages import Message, ToolCall
from nanochat.common import print0


class NemotronReasoningDataset:
    """
    Dataset wrapper for Nemotron reasoning data.
    
    Handles loading and converting Nemotron datasets to Nanochat message format.
    Supports mixing reasoning_on (with explicit CoT) and reasoning_off (direct answers).
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        filter_by: Optional[Dict[str, Any]] = None,
        streaming: bool = False,
        reasoning_ratio: float = 0.7,
        seed: int = 42,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name
                - "nvidia/Nemotron-Post-Training-Dataset-v2"
                - "nvidia/Nemotron-Cascade-SFT-Stage-2"
                - "nvidia/Nemotron-Instruction-Following-Chat-v1"
            split: Dataset split (default: "train")
            subset: Dataset subset/configuration (e.g., "SFT" for Post-Training-v2)
            filter_by: Dict of field:value pairs to filter by (e.g., {"category": "math"})
            streaming: Whether to use streaming mode (for very large datasets)
            reasoning_ratio: Ratio of reasoning_on to total examples (0.0 to 1.0)
                - 0.7 means 70% reasoning_on, 30% reasoning_off
            seed: Random seed for deterministic sampling
        """
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.filter_by = filter_by or {}
        self.streaming = streaming
        self.reasoning_ratio = reasoning_ratio
        self.seed = seed
        self._category_handled_by_config = False
        
        filter_desc = f", filter={filter_by}" if filter_by else ""
        print0(f"Loading {dataset_name} (subset={subset}, split={split}{filter_desc})...")
        
        # Load dataset from HuggingFace
        try:
            # Special handling for Nemotron-Cascade-SFT-Stage-2
            # This dataset requires the category as a config name
            if dataset_name == "nvidia/Nemotron-Cascade-SFT-Stage-2" and filter_by and "category" in filter_by:
                # Use category as config/subset
                cascade_config = filter_by["category"]
                print0(f"  Using config '{cascade_config}' for Cascade dataset")
                self.dataset = load_dataset(
                    dataset_name,
                    cascade_config,
                    split=split,
                    streaming=streaming,
                )
                # Keep filter_by intact for API compatibility, but track which filters were applied
                self._category_handled_by_config = True
            elif subset:
                self.dataset = load_dataset(
                    dataset_name,
                    subset,
                    split=split,
                    streaming=streaming,
                )
            else:
                self.dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=streaming,
                )
            
            # Apply any remaining filters if specified
            if self.filter_by and not streaming:
                # Skip category if it was handled by config
                filters_to_apply = {k: v for k, v in self.filter_by.items() 
                                   if not (k == "category" and self._category_handled_by_config)}
                
                for field, value in filters_to_apply.items():
                    print0(f"  Filtering by {field}={value}...")
                    self.dataset = self.dataset.filter(
                        lambda x: x.get(field) == value,
                        desc=f"Filtering {field}={value}"
                    )
            
            # Get dataset size
            if not streaming:
                self.size = len(self.dataset)
                print0(f"✓ Loaded {self.size:,} examples from {dataset_name}")
            else:
                self.size = -1  # Unknown for streaming
                print0(f"✓ Loaded streaming dataset from {dataset_name}")
                
        except Exception as e:
            print0(f"✗ Failed to load {dataset_name}: {e}")
            raise
        
        self.rng = random.Random(seed)
    
    def __len__(self) -> int:
        """Return dataset size (or -1 for streaming)"""
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example by index"""
        if self.streaming:
            raise NotImplementedError("Indexing not supported for streaming datasets")
        return self._convert_to_nanochat_format(self.dataset[idx])
    
    def _convert_to_nanochat_format(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Nemotron format to Nanochat message format.
        
        Handles four dataset formats:
        
        1. Cascade-SFT-Stage-2:
            {
                "messages": [{"role": "user", "content": "..."}, ...],
                "thinking": true/false,  # Whether this has reasoning
                "category": "math"/"code"/etc,
                "source": "OpenMathReasoning"/etc,
                ...
            }
        
        2. Post-Training-Dataset-v2:
            {
                "messages": [{"role": "user", "content": "..."}, ...],
                ...
            }
        
        3. Instruction-Following-Chat-v1:
            {
                "messages": [
                    {"role": "user", "content": "...", "reasoning_content": null},
                    {"role": "assistant", "content": "...", "reasoning_content": "..."}
                ],
                "reasoning": "on"/"off",  # Whether conversation uses reasoning
                "capability_target": "chat"/"instruction_following",
                ...
            }
        
        4. Dolci-Instruct-SFT-Tool-Use (allenai):
            {
                "messages": [
                    {"role": "system", "content": "...", "functions": "[...]", "function_calls": null},
                    {"role": "user", "content": "...", "functions": null, "function_calls": null},
                    {"role": "assistant", "content": null, "functions": null, "function_calls": "func(...)"},
                    {"role": "environment", "content": "{results}", "functions": null, "function_calls": null},
                    {"role": "assistant", "content": "...", "functions": null, "function_calls": null}
                ],
                "dataset_source": "allenai/...",
                "id": "..."
            }
        
        Nanochat format:
            {
                "messages": [
                    {"role": "user", "content": "..."},
                    {
                        "role": "assistant",
                        "reasoning": "...",  # Extracted from thinking or reasoning_content
                        "reasoning_level": "medium",  # Based on length/complexity
                        "content": "..."  # Final answer
                    }
                ]
            }
        """
        messages = example.get("messages", [])
        
        # Determine if this example has reasoning
        has_thinking = False
        if "thinking" in example:
            # Cascade-SFT-Stage-2 format
            has_thinking = example["thinking"]
        elif "reasoning" in example:
            # Nemotron-SFT-Instruction-Following-Chat-v2 or v1 format
            has_thinking = example["reasoning"] == "on"
        
        # Detect Dolci format (allenai datasets with function_calls/functions fields)
        is_dolci_format = any("function_calls" in msg or "functions" in msg for msg in messages)
        
        # Convert messages to Nanochat format
        converted_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "") or ""  # Handle None content
            
            # Handle Dolci-specific roles
            if role == "environment":
                # Environment role in Dolci = tool results
                # Convert to 'tool' role for nanochat compatibility
                converted_messages.append({
                    "role": "user",  # Treat as user message with tool results
                    "content": f"[Tool Results]\n{content}"
                })
                continue
            
            if role == "user":
                # User messages are straightforward
                converted_messages.append({
                    "role": "user",
                    "content": content
                })
            
            elif role == "assistant":
                # Assistant messages need reasoning structure
                
                # Check for Dolci function_calls format
                function_calls_str = msg.get("function_calls")
                
                # Check for explicit reasoning_content (Instruction-Following-Chat-v1)
                reasoning_content = msg.get("reasoning_content")
                
                # For Dolci tool-use: treat function calls as reasoning
                if is_dolci_format and function_calls_str:
                    # Assistant is making a tool call
                    # Treat the function call as "reasoning" about which tool to use
                    reasoning_text = f"I need to use tools to answer this question."
                    answer_text = function_calls_str  # The actual function calls
                    
                    converted_messages.append({
                        "role": "assistant",
                        "reasoning": reasoning_text,
                        "reasoning_level": "low",
                        "content": answer_text
                    })
                    continue
                
                # Determine if we should use reasoning for this example
                # Based on reasoning_ratio and whether the example has thinking
                use_reasoning = has_thinking and self.rng.random() < self.reasoning_ratio
                
                if use_reasoning:
                    # Parse content to extract reasoning and answer
                    if reasoning_content:
                        # Use explicit reasoning_content field
                        reasoning_text = reasoning_content
                        answer_text = content
                    else:
                        # Parse from content
                        reasoning_text, answer_text = self._parse_reasoning_content(content)
                    
                    # Determine reasoning level based on length/complexity
                    reasoning_level = self._infer_reasoning_level(reasoning_text)
                    
                    converted_messages.append({
                        "role": "assistant",
                        "reasoning": reasoning_text,
                        "reasoning_level": reasoning_level,
                        "content": answer_text
                    })
                else:
                    # No reasoning mode - use think_none
                    # Extract just the answer (skip reasoning if present)
                    if reasoning_content:
                        # Use only content, skip reasoning_content
                        answer_text = content
                    else:
                        # Parse to extract answer without reasoning
                        _, answer_text = self._parse_reasoning_content(content)
                    
                    converted_messages.append({
                        "role": "assistant",
                        "reasoning_level": "none",
                        "content": answer_text
                    })
            
            elif role == "system":
                # System messages
                # For Dolci: system message contains function definitions
                functions_str = msg.get("functions")
                
                if is_dolci_format and functions_str:
                    # Include both the instruction and function definitions
                    full_content = content
                    if functions_str:
                        # Functions are in JSON string format, make them readable
                        full_content += f"\n\nAvailable functions:\n{functions_str}"
                    
                    converted_messages.append({
                        "role": "system",
                        "content": full_content
                    })
                else:
                    converted_messages.append({
                        "role": "system",
                        "content": content
                    })
            
            # Note: Tool messages would be handled here if present in dataset
        
        return {
            "messages": converted_messages,
            "metadata": {
                "source": example.get("source", example.get("dataset_source", "unknown")),
                "category": example.get("category", example.get("capability_target", "tool_use" if is_dolci_format else "unknown")),
                "had_thinking": has_thinking,
                "is_tool_use": is_dolci_format,
            }
        }
    
    def _parse_reasoning_content(self, content: str) -> Tuple[str, str]:
        """
        Parse assistant content to extract reasoning and answer.
        
        Common patterns in Nemotron datasets:
        1. <think>reasoning...</think>\nAnswer: ...
        2. Reasoning text followed by final answer in boxed format
        3. Step-by-step solution ending with final answer
        
        Returns:
            (reasoning_text, answer_text)
        """
        # Pattern 1: Explicit <think> tags
        if "<think>" in content and "</think>" in content:
            think_start = content.index("<think>") + 7
            think_end = content.index("</think>")
            reasoning = content[think_start:think_end].strip()
            answer = content[think_end + 8:].strip()
            return reasoning, answer
        
        # Pattern 2: Look for "Answer:" or similar markers
        answer_markers = [
            "\n\nAnswer:",
            "\nAnswer:",
            "\n\nFinal Answer:",
            "\nFinal Answer:",
            "\\boxed{",
        ]
        
        for marker in answer_markers:
            if marker in content:
                idx = content.index(marker)
                reasoning = content[:idx].strip()
                answer = content[idx:].strip()
                return reasoning, answer
        
        # Pattern 3: No clear separation - use heuristic
        # If content is long (>500 chars), assume first 70% is reasoning
        if len(content) > 500:
            split_point = int(len(content) * 0.7)
            # Try to split at sentence boundary
            for i in range(split_point, min(len(content), split_point + 100)):
                if content[i] in '.!?':
                    reasoning = content[:i+1].strip()
                    answer = content[i+1:].strip()
                    return reasoning, answer
            
            # Fallback to arbitrary split
            reasoning = content[:split_point].strip()
            answer = content[split_point:].strip()
            return reasoning, answer
        
        # Default: No reasoning, entire content is answer
        return "", content
    
    def _infer_reasoning_level(self, reasoning_text: str) -> str:
        """
        Infer reasoning level based on content length and complexity.
        
        Heuristics:
        - none: No reasoning (empty)
        - low: Short reasoning (<200 chars)
        - medium: Moderate reasoning (200-800 chars)
        - high: Long/complex reasoning (>800 chars)
        """
        if not reasoning_text:
            return "none"
        
        length = len(reasoning_text)
        
        if length < 200:
            return "low"
        elif length < 800:
            return "medium"
        else:
            return "high"
    
    def get_iterator(self, shuffle: bool = True) -> Iterator[Dict[str, Any]]:
        """
        Get an iterator over the dataset.
        
        Args:
            shuffle: Whether to shuffle the dataset (only for non-streaming)
        
        Yields:
            Converted examples in Nanochat format
        """
        if self.streaming:
            # Streaming mode - iterate directly
            for example in self.dataset:
                yield self._convert_to_nanochat_format(example)
        else:
            # Non-streaming mode - can shuffle
            indices = list(range(self.size))
            if shuffle:
                self.rng.shuffle(indices)
            
            for idx in indices:
                yield self.__getitem__(idx)


class MixedReasoningDataLoader:
    """
    Mixed dataloader combining multiple reasoning datasets.
    
    Supports:
    - Multiple datasets with different weights
    - Round-robin or weighted sampling
    - Configurable reasoning ratios per dataset
    """
    
    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        seed: int = 42,
    ):
        """
        Args:
            datasets: List of dataset configurations:
                [
                    {
                        "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                        "subset": None,
                        "split": "train",
                        "filter_by": {"category": "math"},
                        "weight": 0.3,
                        "reasoning_ratio": 0.7,
                    },
                    {
                        "name": "nvidia/Nemotron-Post-Training-Dataset-v2",
                        "subset": "SFT",
                        "split": "math",
                        "weight": 0.7,
                        "reasoning_ratio": 0.5,
                    },
                ]
            seed: Random seed
        """
        self.datasets = []
        self.weights = []
        
        for config in datasets:
            dataset = NemotronReasoningDataset(
                dataset_name=config["name"],
                split=config.get("split", "train"),
                subset=config.get("subset"),
                filter_by=config.get("filter_by"),
                streaming=config.get("streaming", False),
                reasoning_ratio=config.get("reasoning_ratio", 0.7),
                seed=seed,
            )
            
            self.datasets.append(dataset)
            self.weights.append(config.get("weight", 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.rng = random.Random(seed)
        
        print0(f"\n✓ Mixed dataloader created with {len(self.datasets)} datasets")
        for i, (dataset, weight) in enumerate(zip(self.datasets, self.weights)):
            filter_desc = f", filter={dataset.filter_by}" if dataset.filter_by else ""
            print0(f"  {i+1}. {dataset.dataset_name}{filter_desc} (weight={weight:.2f})")
    
    def get_iterator(self, shuffle: bool = True) -> Iterator[Dict[str, Any]]:
        """
        Get an iterator that samples from all datasets according to weights.
        
        Args:
            shuffle: Whether to shuffle datasets
        
        Yields:
            Examples from mixed datasets
        """
        # Create iterators for all datasets
        iterators = [ds.get_iterator(shuffle=shuffle) for ds in self.datasets]
        
        # Sample from datasets according to weights
        while True:
            # Choose dataset based on weights
            dataset_idx = self.rng.choices(
                range(len(self.datasets)),
                weights=self.weights,
                k=1
            )[0]
            
            try:
                example = next(iterators[dataset_idx])
                yield example
            except StopIteration:
                # Dataset exhausted - stop or recreate iterator
                # For now, stop (will be handled by training loop)
                break


def create_reasoning_dataloader(
    stage: str = "instruction_following",
    reasoning_ratio: float = 0.7,
    seed: int = 42,
) -> MixedReasoningDataLoader:
    """
    Create a reasoning dataloader for a specific training stage.
    
    Stages:
        1. "instruction_following": Focus on following instructions properly
           - Uses Nemotron-Instruction-Following-Chat-v1 dataset
           
        2. "reasoning": Focus on reasoning capabilities
           - Uses Nemotron-Cascade-SFT-Stage-2 (filtered by category)
           - Uses Nemotron-Post-Training-Dataset-v2 (STEM splits)
           - Mixes reasoning_on and reasoning_off examples
           
        3. "mixed": Combined instruction + reasoning + chat
           - Uses all datasets for comprehensive training
    
    Args:
        stage: Training stage name
        reasoning_ratio: Ratio of reasoning_on to total examples
        seed: Random seed
    
    Returns:
        Configured MixedReasoningDataLoader
    """
    
    if stage == "instruction_following":
        # Stage 1: Instruction Following
        # Use the official Nemotron-SFT-Instruction-Following-Chat-v2 dataset
        # Use streaming mode to avoid downloading 6GB+ files
        datasets = [
            {
                "name": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
                "subset": None,  # Default subset
                "split": "reasoning_on",  # Split with reasoning traces
                "streaming": True,  # Use streaming to avoid disk space issues
                "weight": 0.5,
                "reasoning_ratio": 0.9,  # High ratio for reasoning_on split
            },
            {
                "name": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
                "subset": None,
                "split": "reasoning_off",  # Split without reasoning traces
                "streaming": True,  # Use streaming
                "weight": 0.5,
                "reasoning_ratio": 0.1,  # Low ratio for reasoning_off split
            }
        ]
    
    elif stage == "reasoning":
        # Stage 2: Reasoning Training
        # Use Cascade-SFT-Stage-2 categories as subsets
        # Cascade has configs (subsets): math, code, science, general, tool_calling, instruction-following, swe_*
        # Note: Post-Training-Dataset-v2 is gated - omitted for now
        datasets = [
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "math",  # Math reasoning config
                "split": "train",
                "streaming": True,  # Use streaming to avoid disk issues
                "weight": 0.30,
                "reasoning_ratio": reasoning_ratio,
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "code",  # Code reasoning config
                "split": "train",
                "streaming": True,
                "weight": 0.30,
                "reasoning_ratio": reasoning_ratio,
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "science",  # Science reasoning config
                "split": "train",
                "streaming": True,
                "weight": 0.20,
                "reasoning_ratio": reasoning_ratio,
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "general",  # General reasoning config
                "split": "train",
                "streaming": True,
                "weight": 0.20,
                "reasoning_ratio": reasoning_ratio,
            }
        ]
    
    elif stage == "mixed":
        # Stage 3: Mixed training (instruction + reasoning + chat)
        # Note: Post-Training-Dataset-v2 is gated - using Cascade's instruction-following and general instead
        datasets = [
            {
                "name": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
                "subset": None,
                "split": "reasoning_on",  # With reasoning
                "streaming": True,
                "weight": 0.20,
                "reasoning_ratio": 0.8,
            },
            {
                "name": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
                "subset": None,
                "split": "reasoning_off",  # Without reasoning
                "streaming": True,
                "weight": 0.15,
                "reasoning_ratio": 0.2,
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "general",  # General domain config
                "split": "train",
                "streaming": True,
                "weight": 0.20,
                "reasoning_ratio": 0.5,
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "math",  # Math config
                "split": "train",
                "streaming": True,
                "weight": 0.15,
                "reasoning_ratio": 0.6,
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "code",  # Code config
                "split": "train",
                "streaming": True,
                "weight": 0.15,
                "reasoning_ratio": 0.6,
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "instruction-following",  # Instruction following config
                "split": "train",
                "streaming": True,
                "weight": 0.15,
                "reasoning_ratio": 0.5,
            }
        ]
    
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    return MixedReasoningDataLoader(datasets, seed=seed)


if __name__ == "__main__":
    """Test the reasoning dataloader"""
    print("Testing Reasoning Dataloader")
    print("=" * 60)
    
    # Test 1: Load a small sample from Cascade dataset
    print("\n1. Testing single dataset loader...")
    try:
        dataset = NemotronReasoningDataset(
            dataset_name="nvidia/Nemotron-Cascade-SFT-Stage-2",
            split="train",
            streaming=True,  # Use streaming for testing
            reasoning_ratio=0.7,
        )
        
        # Get a few examples
        iterator = dataset.get_iterator(shuffle=False)
        for i, example in enumerate(iterator):
            if i >= 3:
                break
            print(f"\n  Example {i+1}:")
            print(f"    Messages: {len(example['messages'])}")
            print(f"    Source: {example['metadata']['source']}")
            print(f"    Category: {example['metadata']['category']}")
            
            # Show first assistant message
            for msg in example['messages']:
                if msg['role'] == 'assistant':
                    print(f"    Reasoning level: {msg.get('reasoning_level', 'none')}")
                    if msg.get('reasoning'):
                        print(f"    Reasoning length: {len(msg['reasoning'])} chars")
                    break
        
        print("\n  ✓ Single dataset loader works!")
    
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
    
    # Test 2: Mixed dataloader
    print("\n2. Testing mixed dataloader...")
    try:
        loader = create_reasoning_dataloader(
            stage="instruction_following",
            reasoning_ratio=0.7,
        )
        
        print("  ✓ Mixed dataloader created!")
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed!")
