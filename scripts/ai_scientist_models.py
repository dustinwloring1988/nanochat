import argparse
import json

from ai_scientist.providers import list_models, llm_budget, preflight_model


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect AI Scientist model providers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--model", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--model", required=True)
    subparsers.add_parser("budget")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "list":
        result = list_models(args.model)
    elif args.command == "preflight":
        result = preflight_model(args.model)
    else:
        result = llm_budget.snapshot()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
