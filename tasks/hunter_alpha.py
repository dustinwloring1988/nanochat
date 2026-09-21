"""
Hunter Alpha Coding Agent SFT by heegyu.
https://huggingface.co/datasets/heegyu/Hunter-Alpha-Coding-Agent-SFT

1.2K examples of tool-using coding agent trajectories.
Includes 'tools' field with function definitions and multi-turn tool use.
"""

from tasks.common import Task, load_hub_dataset

class HunterAlpha(Task):
    """Hunter Alpha coding agent dataset. 1.2K tool-using agent examples."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        # This dataset only has train split
        assert split in ["train"], "HunterAlpha only has train split"
        self.ds = load_hub_dataset("heegyu/Hunter-Alpha-Coding-Agent-SFT", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        tools = row.get("tools", [])  # Tool definitions
        
        # Sanity checks
        assert len(messages) >= 2, "Must have at least 2 messages"
        
        # Check for system message
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        
        # Verify alternating structure (user/assistant/tool results)
        # Note: This dataset has tool use, so roles can be user/assistant/tool
        assert len(rest_messages) >= 2, "Must have at least user+assistant"
        
        # Build conversation with tools
        conversation = {"messages": messages}
        if tools:
            conversation["tools"] = tools
        
        return conversation
