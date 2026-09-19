"""
Reasoning Model Configuration

This configuration defines the training parameters for reasoning-capable models
using NVIDIA Nemotron datasets and Allen AI Dolci datasets.

Training Philosophy:
    - Stage 1: Instruction following (30% of training)
    - Stage 2: Reasoning training with CoT + Tool Use (50% of training)
    - Stage 3: Multi-task fine-tuning (20% of training)

Dataset Sources:
    - nvidia/Nemotron-Cascade-SFT-Stage-2: 7.8M examples
      (math, code, science, tool calling, SWE, general, instruction following)
      NOTE: Single config, filter by 'category' field, use 'thinking' field for reasoning on/off
    
    - nvidia/Nemotron-Post-Training-Dataset-v2: 5.3M examples 
      (multilingual reasoning across math, code, stem, chat)
      NOTE: Config="SFT", splits are categories (math, code, stem, chat, multilingual_*)
    
    - nvidia/Nemotron-Instruction-Following-Chat-v1: 431K examples
      (chat and instruction following with reasoning on/off modes)
      NOTE: Use 'reasoning' field ("on"/"off"), 'reasoning_content' in messages
    
    - allenai/Dolci-Instruct-SFT-Tool-Use: ~1M+ examples
      (function calling and tool use training)
      NOTE: Messages have 'function_calls' and 'functions' fields, 'environment' role for results

Reasoning Modes:
    - reasoning_on: Model generates explicit chain-of-thought before answer
    - reasoning_off: Model provides direct answer without reasoning trace
    - Mixed training enables the model to adapt based on task complexity
"""

CONFIG = {
    # Model configuration (inherited from base pretraining)
    "model": {
        "sequence_len": 2048,
        "vocab_size": 32768,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    },
    
    # Training configuration
    "training": {
        # Total training iterations
        "total_iterations": 2000,
        
        # Stage durations (as fractions of total)
        "stage1_fraction": 0.30,  # Instruction following
        "stage2_fraction": 0.50,  # Reasoning training
        "stage3_fraction": 0.20,  # Multi-task fine-tuning
        
        # Batch configuration
        "batch_size": 4,  # Per GPU (device-batch-size)
        "total_batch_size": 524288,  # Total tokens per update
        "max_seq_len": 2048,  # Maximum sequence length
        
        # Optimization
        "learning_rate": {
            "embedding": 0.3,  # Adam for embeddings
            "unembedding": 0.004,  # Adam for unembedding
            "matrix": 0.02,  # Muon for matrix parameters
        },
        "init_lr_frac": 0.8,  # Initial LR as fraction of base
        "warmup_ratio": 0.05,  # 5% warmup
        "warmdown_ratio": 0.1,  # 10% warmdown
        "final_lr_frac": 0.0,  # LR at end of training
        
        # Precision
        "mixed_precision": "bf16",  # or "fp16" or "fp32"
        
        # Evaluation
        "eval_every": 200,
        "eval_tokens": 40 * 524288,
        "chatcore_every": 200,
    },
    
    # Reasoning configuration
    "reasoning": {
        # Reasoning ratio: fraction of examples with explicit CoT
        # 0.0 = all direct answers, 1.0 = all reasoning traces
        "base_ratio": 0.7,
        
        # Curriculum learning: gradually increase reasoning ratio
        "enable_curriculum": True,
        "curriculum_start": 0.3,  # Start at 30% reasoning
        "curriculum_end": 0.7,    # Ramp up to 70% reasoning
        
        # Reasoning level distribution
        "level_distribution": {
            "none": 0.30,    # 30% no reasoning
            "low": 0.20,     # 20% short reasoning
            "medium": 0.35,  # 35% moderate reasoning
            "high": 0.15,    # 15% long/complex reasoning
        },
        
        # Reasoning inference heuristics
        "auto_level_threshold": {
            "low": 200,     # < 200 chars
            "medium": 800,  # 200-800 chars
            "high": 800,    # > 800 chars
        },
    },
    
    # Dataset configuration
    "datasets": {
        # Stage 1: Instruction Following
        "stage1": [
            {
                "name": "nvidia/Nemotron-Instruction-Following-Chat-v1",
                "subset": None,
                "split": "train",
                "filter_by": {"capability_target": "instruction_following"},
                "weight": 0.7,
                "reasoning_ratio": 0.3,  # Lower ratio for instruction following
                "description": "Verified instruction following with IFEval/IFBench"
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": None,
                "split": "train",
                "filter_by": {"category": "instruction_following"},
                "weight": 0.3,
                "reasoning_ratio": 0.3,
                "description": "Instruction following from Cascade dataset"
            }
        ],
        
        # Stage 2: Reasoning Training
        "stage2": [
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": None,
                "split": "train",
                "filter_by": {"category": "math"},
                "weight": 0.20,
                "reasoning_ratio": 0.7,  # High ratio for reasoning
                "description": "Math reasoning (OpenMathReasoning)"
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": None,
                "split": "train",
                "filter_by": {"category": "code"},
                "weight": 0.20,
                "reasoning_ratio": 0.7,
                "description": "Code reasoning (OpenCodeReasoning, TACO)"
            },
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": None,
                "split": "train",
                "filter_by": {"category": "science"},
                "weight": 0.10,
                "reasoning_ratio": 0.7,
                "description": "Science reasoning"
            },
            {
                "name": "allenai/Dolci-Instruct-SFT-Tool-Use",
                "subset": None,
                "split": "train",
                "weight": 0.15,
                "reasoning_ratio": 0.8,  # High ratio - tool use requires reasoning
                "description": "Tool-use and function calling training"
            },
            {
                "name": "nvidia/Nemotron-Post-Training-Dataset-v2",
                "subset": "SFT",
                "split": "math",
                "weight": 0.15,
                "reasoning_ratio": 0.7,
                "description": "Multilingual math reasoning"
            },
            {
                "name": "nvidia/Nemotron-Post-Training-Dataset-v2",
                "subset": "SFT",
                "split": "code",
                "weight": 0.10,
                "reasoning_ratio": 0.7,
                "description": "Multilingual code reasoning"
            },
            {
                "name": "nvidia/Nemotron-Post-Training-Dataset-v2",
                "subset": "SFT",
                "split": "stem",
                "weight": 0.10,
                "reasoning_ratio": 0.7,
                "description": "STEM reasoning"
            }
        ],
        
        # Stage 3: Mixed (includes legacy tasks)
        "stage3": {
            "reasoning_datasets": [
                {
                    "name": "nvidia/Nemotron-Instruction-Following-Chat-v1",
                    "subset": None,
                    "split": "train",
                    "filter_by": {"capability_target": "chat"},
                    "weight": 0.20,
                    "reasoning_ratio": 0.5,
                },
                {
                    "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                    "subset": None,
                    "split": "train",
                    "filter_by": {"category": "general"},
                    "weight": 0.20,
                    "reasoning_ratio": 0.5,
                },
                {
                    "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                    "subset": None,
                    "split": "train",
                    "filter_by": {"category": "math"},
                    "weight": 0.15,
                    "reasoning_ratio": 0.6,
                },
                {
                    "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                    "subset": None,
                    "split": "train",
                    "filter_by": {"category": "code"},
                    "weight": 0.10,
                    "reasoning_ratio": 0.6,
                },
                {
                    "name": "allenai/Dolci-Instruct-SFT-Tool-Use",
                    "subset": None,
                    "split": "train",
                    "weight": 0.15,
                    "reasoning_ratio": 0.7,
                    "description": "Tool-use and function calling"
                },
                {
                    "name": "nvidia/Nemotron-Post-Training-Dataset-v2",
                    "subset": "SFT",
                    "split": "chat",
                    "weight": 0.20,
                    "reasoning_ratio": 0.5,
                }
            ],
            "legacy_tasks": {
                "smoltalk_epochs": 1,
                "mmlu_epochs": 1,
                "gsm8k_epochs": 2,
            }
        }
    },
    
    # Data processing
    "data": {
        "num_workers": 4,
        "prefetch_factor": 2,
        "seed": 42,
        "shuffle": True,
        "streaming": False,  # Set to True for very large datasets
    },
    
    # Output
    "output": {
        "run_name": "reasoning-sft",
        "checkpoint_dir": "./reasoning_sft_checkpoints",
        "save_every": 500,
        "log_every": 10,
    },
    
    # Evaluation tasks (for ChatCORE)
    "evaluation": {
        "tasks": ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"],
        "categorical_tasks": ["ARC-Easy", "ARC-Challenge", "MMLU"],
        "max_categorical_problems": -1,  # -1 = no limit
        "max_generative_problems": 24,
        "baseline_accuracies": {
            "ARC-Easy": 0.25,
            "ARC-Challenge": 0.25,
            "MMLU": 0.25,
            "GSM8K": 0.0,
            "HumanEval": 0.0,
        }
    }
}

# Precompute stage boundaries
def get_stage_boundaries(total_iterations):
    """Calculate iteration boundaries for each stage"""
    stage1_end = int(CONFIG["training"]["total_iterations"] * CONFIG["training"]["stage1_fraction"])
    stage2_end = stage1_end + int(CONFIG["training"]["total_iterations"] * CONFIG["training"]["stage2_fraction"])
    stage3_end = CONFIG["training"]["total_iterations"]
    
    return {
        "stage1": (0, stage1_end),
        "stage2": (stage1_end, stage2_end),
        "stage3": (stage2_end, stage3_end),
    }

# Helper functions
def get_stage_from_step(step):
    """Determine current training stage from step number"""
    boundaries = get_stage_boundaries(CONFIG["training"]["total_iterations"])
    
    if step < boundaries["stage1"][1]:
        return "instruction_following"
    elif step < boundaries["stage2"][1]:
        return "reasoning"
    else:
        return "mixed"

def get_reasoning_ratio_for_step(step):
    """Get reasoning ratio for current step (with curriculum)"""
    config = CONFIG["reasoning"]
    
    if not config["enable_curriculum"]:
        return config["base_ratio"]
    
    boundaries = get_stage_boundaries(CONFIG["training"]["total_iterations"])
    stage = get_stage_from_step(step)
    
    if stage == "instruction_following":
        # Stage 1: Low reasoning
        return config["curriculum_start"]
    
    elif stage == "reasoning":
        # Stage 2: Linear ramp from start to end
        stage_start, stage_end = boundaries["stage2"]
        progress = (step - stage_start) / (stage_end - stage_start)
        return config["curriculum_start"] + progress * (config["curriculum_end"] - config["curriculum_start"])
    
    else:
        # Stage 3: Full reasoning capability
        return config["base_ratio"]

# Export for easy access
STAGE_BOUNDARIES = get_stage_boundaries(CONFIG["training"]["total_iterations"])

if __name__ == "__main__":
    """Test configuration and helper functions"""
    print("Reasoning Model Configuration")
    print("=" * 60)
    
    print("\nTraining Schedule:")
    print(f"  Total iterations: {CONFIG['training']['total_iterations']}")
    print(f"  Stage 1 (Instruction): {STAGE_BOUNDARIES['stage1']}")
    print(f"  Stage 2 (Reasoning):   {STAGE_BOUNDARIES['stage2']}")
    print(f"  Stage 3 (Mixed):       {STAGE_BOUNDARIES['stage3']}")
    
    print("\nReasoning Curriculum:")
    test_steps = [0, 300, 600, 1000, 1500, 2000]
    for step in test_steps:
        stage = get_stage_from_step(step)
        ratio = get_reasoning_ratio_for_step(step)
        print(f"  Step {step:4d}: stage={stage:20s} reasoning_ratio={ratio:.2f}")
    
    print("\nDataset Weights:")
    print("  Stage 1:", CONFIG["datasets"]["stage1"])
    print("  Stage 2:", CONFIG["datasets"]["stage2"])
    print("  Stage 3:", CONFIG["datasets"]["stage3"])
    
    print("\n✓ Configuration validated!")
