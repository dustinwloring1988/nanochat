"""
Stage 2: Long-context math & reasoning pretraining configuration
Context: 8192 tokens (for Math-Textbooks) or 32768 tokens (for InfiniByte-Reasoning)
Datasets: Math-Textbooks + InfiniByte-Reasoning
"""

# Stage 2a: Math Textbooks (8k context)
CONFIG_STAGE2A = {
    # Model configuration
    "model": {
        "sequence_len": 8192,
        "vocab_size": 32768,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    },
    
    # Training configuration
    "training": {
        "batch_size": 2,  # Reduced for longer context
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "weight_decay": 0.1,
        "warmup_steps": 1000,
        "max_steps": 50000,
        "eval_interval": 1000,
        "save_interval": 5000,
        "mixed_precision": "bf16",
    },
    
    # Dataset configuration
    "datasets": [
        {
            "name": "nvidia/Nemotron-Pretraining-Specialized-v1",
            "subset": "Nemotron-Pretraining-Math-Textbooks",
            "weight": 1.0,
            "description": "Mathematical textbooks and explanations"
        },
    ],
    
    # Data processing
    "data": {
        "num_workers": 4,
        "prefetch_factor": 2,
        "seed": 42,
        "shuffle_buffer_size": 5000,
    },
    
    # Optimization
    "optimizer": {
        "type": "adamw",
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    },
    
    # Output
    "output": {
        "run_name": "nemotron-stage2a-math",
        "checkpoint_dir": "./checkpoints/stage2a",
        "log_dir": "./logs/stage2a",
        "load_from": "./checkpoints/stage1/final.pt",  # Continue from stage 1
    }
}

# Stage 2b: InfiniByte Reasoning (32k context - very long!)
CONFIG_STAGE2B = {
    # Model configuration
    "model": {
        "sequence_len": 32768,
        "vocab_size": 32768,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    },
    
    # Training configuration
    "training": {
        "batch_size": 1,  # Very limited for 32k context
        "gradient_accumulation_steps": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "warmup_steps": 500,
        "max_steps": 20000,
        "eval_interval": 500,
        "save_interval": 2000,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,  # Required for memory
    },
    
    # Dataset configuration
    "datasets": [
        {
            "name": "nvidia/Nemotron-Pretraining-Specialized-v1",
            "subset": "Nemotron-Pretraining-InfiniByte-Reasoning",
            "weight": 1.0,
            "description": "Long-context reasoning problems"
        },
    ],
    
    # Data processing
    "data": {
        "num_workers": 2,
        "prefetch_factor": 1,
        "seed": 42,
        "shuffle_buffer_size": 1000,
    },
    
    # Optimization
    "optimizer": {
        "type": "adamw",
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    },
    
    # Output
    "output": {
        "run_name": "nemotron-stage2b-reasoning",
        "checkpoint_dir": "./checkpoints/stage2b",
        "log_dir": "./logs/stage2b",
        "load_from": "./checkpoints/stage2a/final.pt",  # Continue from stage 2a
    }
}
