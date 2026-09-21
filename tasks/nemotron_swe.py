"""
Nemotron SFT SWE v3.5 by NVIDIA.
https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3.5

5.1K examples of repository-level software engineering agent trajectories.
Includes 'tools' field with bash, editor, and other SWE tools.
"""

from tasks.common import Task, load_hub_dataset

class NemotronSWE(Task):
    """Nemotron SFT SWE v3.5 dataset. 5.1K software engineering agent examples."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        # This dataset only has train split
        assert split in ["train"], "NemotronSWE only has train split"
        self.ds = load_hub_dataset("nvidia/Nemotron-SFT-SWE-v3.5", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        tools = row.get("tools", [])  # SWE tool definitions (bash, editor, etc.)
        
        # Sanity checks
        assert len(messages) >= 2, "Must have at least 2 messages"
        
        # Check for system message
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        
        assert len(rest_messages) >= 2, "Must have at least user+assistant"
        
        # Build conversation with tools
        conversation = {"messages": messages}
        if tools:
            conversation["tools"] = tools
        
        return conversation
