from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_suite.data import prepare_loveda


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LoveDA Train.zip and Val.zip only")
    parser.add_argument("--zip-root", type=Path, default=Path(r"D:\landcover\data\loveda"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\landcover\data\loveda\extracted"))
    args = parser.parse_args()
    path = prepare_loveda(args.zip_root, args.output)
    print(f"LoveDA Train/Val prepared at: {path}")
    print("Test.zip was not read or extracted.")


if __name__ == "__main__":
    main()
