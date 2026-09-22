"""
Nemotron SFT Multilingual v2 by NVIDIA.
https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2

370K examples across Hindi, Korean, Portuguese, Japanese with math, code, and STEM reasoning.
Includes optional 'tools' field for function calling.

Note: This dataset has 12 language-domain splits (code_hi, math_ja, stem_ko, etc.)
For full training, you'd want to load all splits. For now, we load code_hi as a representative sample.
"""

from tasks.common import Task, load_hub_dataset, normalize_messages

class NemotronMultilingual(Task):
    """Nemotron SFT Multilingual v2 dataset. 370K multilingual instruction following examples."""

    def __init__(self, split="code_hi", **kwargs):
        super().__init__(**kwargs)
        valid_splits = ["code_hi", "code_ja", "code_ko", "code_pt",
                       "math_hi", "math_ja", "math_ko", "math_pt",
                       "stem_hi", "stem_ja", "stem_ko", "stem_pt"]
        assert split in valid_splits, f"Split must be one of {valid_splits}"
        self.ds = load_hub_dataset("nvidia/Nemotron-SFT-Multilingual-v2", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        import json
        row = self.ds[index]
        messages = normalize_messages(row.get("messages", []))
        
        # Handle tools field
        tools = row.get("tools", [])
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except Exception:
                tools = []
        
        conversation = {"messages": messages}
        if tools and isinstance(tools, list):
            conversation["tools"] = tools
        return conversation
