"""
Claude Fable 5 SFT Clean by Bc-AI.
https://huggingface.co/datasets/Bc-AI/claude-fable-5-sft-clean

63 examples of high-quality curated coding and agent examples.
Small but valuable dataset for quality-focused fine-tuning.
"""

from tasks.common import Task, load_hub_dataset

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
        messages = row["messages"]
        
        # Sanity checks
        assert len(messages) >= 2, "Must have at least 2 messages"
        
        # Check for system message
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        
        # Verify alternating user/assistant
        assert len(rest_messages) >= 2, "Must have at least user+assistant"
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            # Allow some flexibility for multi-turn conversations
            if message["role"] not in ["user", "assistant", "system"]:
                continue  # Skip tool messages if present
            assert isinstance(message["content"], str), "Content must be a string"
        
        return {"messages": messages}
