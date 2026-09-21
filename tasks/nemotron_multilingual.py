"""
Nemotron SFT Multilingual v2 by NVIDIA.
https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2

370K examples across Hindi, Korean, Portuguese, Japanese with math, code, and STEM reasoning.
Includes optional 'tools' field for function calling.

Note: This dataset has 12 language-domain splits (code_hi, math_ja, stem_ko, etc.)
For full training, you'd want to load all splits. For now, we load code_hi as a representative sample.
"""

from tasks.common import Task, load_hub_dataset

class NemotronMultilingual(Task):
    """Nemotron SFT Multilingual v2 dataset. 370K multilingual instruction following examples."""

    def __init__(self, split="code_hi", **kwargs):
        super().__init__(**kwargs)
        # This dataset has 12 language-domain splits:
        # code_hi, code_ja, code_ko, code_pt (Hindi, Japanese, Korean, Portuguese code)
        # math_hi, math_ja, math_ko, math_pt (multilingual math)
        # stem_hi, stem_ja, stem_ko, stem_pt (multilingual STEM)
        # For full curriculum, load all 12 and concat. For testing, we use code_hi.
        valid_splits = ["code_hi", "code_ja", "code_ko", "code_pt",
                       "math_hi", "math_ja", "math_ko", "math_pt",
                       "stem_hi", "stem_ja", "stem_ko", "stem_pt"]
        assert split in valid_splits, f"Split must be one of {valid_splits}"
        self.ds = load_hub_dataset("nvidia/Nemotron-SFT-Multilingual-v2", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        tools = row.get("tools", [])  # Optional tools field
        
        # Sanity checks - messages should be a list
        if not isinstance(messages, list):
            # Dataset format issue - skip this example
            return {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
        
        assert len(messages) >= 2, "Must have at least 2 messages (user+assistant)"
        
        # Check if there's a system message
        first_message = messages[0]
        if not isinstance(first_message, dict):
            # Format issue - create minimal valid conversation
            return {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
        
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        
        # Verify alternating user/assistant messages
        assert len(rest_messages) >= 2, "Must have at least user+assistant after optional system"
        for i, message in enumerate(rest_messages):
            if not isinstance(message, dict):
                continue
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} role mismatch"
            assert isinstance(message["content"], str), "Content must be a string"
        
        # Return conversation (include tools if present)
        conversation = {"messages": messages}
        if tools:
            conversation["tools"] = tools
        
        return conversation
