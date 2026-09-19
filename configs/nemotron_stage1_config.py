"""
Stage 1: Code-focused pretraining configuration
Context: 2048 tokens
Datasets: Code-Concepts + Scientific-Coding
"""

CONFIG = {
    # Model configuration
    "model": {
        "sequence_len": 2048,
        "vocab_size": 32768,  # Adjust based on your tokenizer
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    },
    
    # Training configuration
    "training": {
        "batch_size": 8,  # Per GPU
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "warmup_steps": 2000,
        "max_steps": 100000,
        "eval_interval": 1000,
        "save_interval": 5000,
        "mixed_precision": "bf16",
    },
    
    # Dataset configuration
    "datasets": [
        {
            "name": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
            "subset": "Nemotron-Pretraining-Code-Concepts",
            "weight": 0.5,
            "description": "Code concepts and programming patterns"
        },
        {
            "name": "nvidia/Nemotron-Pretraining-Specialized-v1",
            "subset": "Nemotron-Pretraining-Scientific-Coding",
            "weight": 0.5,
            "description": "Scientific computing and algorithmic problem solving"
        },
    ],
    
    # Data processing
    "data": {
        "num_workers": 4,
        "prefetch_factor": 2,
        "seed": 42,
        "shuffle_buffer_size": 10000,
    },
    
    # Optimization
    "optimizer": {
        "type": "adamw",
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    },
    
    # Output
    "output": {
        "run_name": "nemotron-stage1-code",
        "checkpoint_dir": "./checkpoints/stage1",
        "log_dir": "./logs/stage1",
    }
}
