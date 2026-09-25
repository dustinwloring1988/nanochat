"""Legacy SFT entry point retired in favor of the approved fixed-context trainer."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Disabled legacy SFT entry point")
    parser.error(
        "scripts/chat_sft.py is disabled because it writes to the trusted cache; use scripts.sft_train_curriculum.py with --run-dir and --dataset-manifest"
    )


if __name__ == "__main__":
    main()
