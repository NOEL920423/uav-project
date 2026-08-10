"""Validate a BC dataset and print machine-readable statistics."""

import argparse
import json

from uav_ml.datasets.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/bc_v0")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

