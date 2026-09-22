"""
Claude Fable 5 SFT Clean by Bc-AI.
https://huggingface.co/datasets/Bc-AI/claude-fable-5-sft-clean

63 examples of high-quality curated coding and agent examples.
Small but valuable dataset for quality-focused fine-tuning.
"""

from tasks.common import Task, load_hub_dataset, normalize_messages

class ClaudeFable(Task):
    """Claude Fable 5 SFT Clean dataset. 63 high-quality curated examples."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        # This dataset only has train split
        assert split in ["train"], "ClaudeFable only has train split"
        self.ds = load_hub_dataset("Bc-AI/claude-fable-5-sft-clean", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = normalize_messages(row.get("messages", []))
        return {"messages": messages}
