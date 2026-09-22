"""
Nemotron SFT SWE v3.5 by NVIDIA.
https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3.5

5.1K examples of repository-level software engineering agent trajectories.
Includes 'tools' field with bash, editor, and other SWE tools.
"""

from tasks.common import Task, load_hub_dataset, normalize_messages

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
