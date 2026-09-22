"""
Hunter Alpha Coding Agent SFT by heegyu.
https://huggingface.co/datasets/heegyu/Hunter-Alpha-Coding-Agent-SFT

1.2K examples of tool-using coding agent trajectories.
Includes 'tools' field with function definitions and multi-turn tool use.
"""

from tasks.common import Task, load_hub_dataset, normalize_messages

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
        import json
        row = self.ds[index]
        messages = normalize_messages(row.get("messages", []))
        
        # Handle tools field - might be a JSON string, list, or missing
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
