"""Legacy SFT entry point retired in favor of the approved fixed-context trainer."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Disabled legacy SFT entry point")
    parser.error(
        "scripts/chat_sft.py is disabled because it writes to the trusted cache. "
        "The default workflows are pretraining-only. For the separately approved "
        "fixed-context runtime probe, run: python -m scripts.sft_smoke --help"
    )


if __name__ == "__main__":
    main()
