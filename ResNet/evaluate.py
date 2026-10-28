#!/usr/bin/env python3
"""Step 3: evaluate the trained checkpoint on the UNSEEN test split (run once, at the end).

    python evaluate.py --data fakeddit_sample --weights resnet50_fakeddit.pt
"""
import argparse
import json
import sys

from resnet_fake import ResNetFakeError
from resnet_fake.evaluation import evaluate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="fakeddit_sample")
    p.add_argument("--weights", default="resnet50_fakeddit.pt")
    p.add_argument("--split", default="test", choices=["test", "val"], help="use 'val' for threshold/debug work, 'test' only for the final number")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    try:
        report = evaluate(a.weights, f"{a.data}/{a.split}.csv", a.data, threshold=a.threshold)
    except (ResNetFakeError, FileNotFoundError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
