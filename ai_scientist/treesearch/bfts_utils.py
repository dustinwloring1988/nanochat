import os
import os.path as osp
import shutil
from pathlib import Path

import yaml


def idea_to_markdown(data: dict, output_path: str, load_code: str) -> None:
    """
    Convert a dictionary into a markdown file.

    Args:
        data: Dictionary containing the data to convert
        output_path: Path where the markdown file will be saved
        load_code: Path to a code file to include in the markdown
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            # Convert key to title format and make it a header
            header = key.replace("_", " ").title()
            f.write(f"## {header}\n\n")

            # Handle different value types
            if isinstance(value, (list, tuple)):
                for item in value:
                    f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"### {sub_key}\n")
                    f.write(f"{sub_value}\n\n")
            else:
                f.write(f"{value}\n\n")

        # Add the code to the markdown file
        if load_code:
            # Assert that the code file exists before trying to open it
            assert os.path.exists(
                load_code
            ), f"Code path at {load_code} must exist if using the 'load_code' flag. This is an optional code prompt that you may choose to include; if not, please do not set 'load_code'."
            f.write(f"## Code To Potentially Use\n\n")
            f.write(f"Use the following code as context for your experiments:\n\n")
            with open(load_code, "r") as code_file:
                code = code_file.read()
                f.write(f"```python\n{code}\n```\n\n")


def edit_bfts_config_file(
    config_path: str,
    idea_dir: str,
    idea_path: str,
    model: str = None,
    max_nodes: int = 1,
) -> str:
    """
    Edit the bfts_config.yaml file to point to the idea.md file

    Args:
        config_path: Path to the bfts_config.yaml file
        idea_dir: Directory where the idea.md file is located
        idea_path: Path to the idea.md file

    Returns:
        Path to the edited bfts_config.yaml file
    """
    run_config_path = osp.join(idea_dir, "bfts_config.yaml")
    shutil.copy(config_path, run_config_path)
    with open(run_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if model:
        config["report"]["model"] = model
        config["experiment"]["model"] = model
        for role in (
            "code",
            "feedback",
            "vlm_feedback",
            "summary",
            "select_node",
        ):
            role_config = config["agent"].setdefault(
                role, {"temp": 0.2, "max_tokens": 4096}
            )
            role_config["model"] = model
    config["agent"]["max_nodes"] = max_nodes
    config["desc_file"] = idea_path
    config["workspace_dir"] = idea_dir

    # make an empty data directory
    data_dir = osp.join(idea_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    config["data_dir"] = data_dir

    # make an empty log directory
    log_dir = osp.join(idea_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = log_dir

    if config.get("experiment", {}).get("mode") == "nanochat":
        source_dir = os.environ.get("NANOCHAT_SOURCE_ROOT", os.getcwd())
        shared_cache = os.environ.get("NANOCHAT_SHARED_CACHE")
        if not shared_cache:
            raise ValueError(
                "NANOCHAT_SHARED_CACHE is required for nanochat experiments"
            )
        config["experiment"]["source_dir"] = source_dir
        config["experiment"]["cache_dir"] = shared_cache
        config_file = Path(config["experiment"]["config_file"])
        if not config_file.is_absolute():
            config_file = Path(source_dir) / config_file
        config["experiment"]["config_file"] = str(config_file.resolve())

    with open(run_config_path, "w") as f:
        yaml.dump(config, f)
    return run_config_path
