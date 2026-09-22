"""
SmolTalk by HuggingFace. Good "general" conversational dataset.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
We use the "smol" version, which is more appropriate for smaller models.
"""

from tasks.common import Task, load_hub_dataset, normalize_messages

class SmolTalk(Task):
    """ smol-smoltalk dataset. train is 460K rows, test is 24K rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_hub_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = normalize_messages(row.get("messages", []))
        return {"messages": messages}
