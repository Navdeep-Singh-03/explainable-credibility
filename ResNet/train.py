#!/usr/bin/env python3
"""Step 2: train ResNet-50 once and save the best (by VALIDATION AUC) checkpoint.

    python train.py --data fakeddit_sample --out resnet50_fakeddit.pt

Uses only train.csv and val.csv. The test set is not touched here.
"""
import argparse
import sys

from resnet_fake.training import train


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="fakeddit_sample", help="folder written by prepare_data.py")
    p.add_argument("--out", default="resnet50_fakeddit.pt")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2, help="set 0 if Windows data-loader errors occur")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline; much worse accuracy)")
    a = p.parse_args()
    try:
        train(f"{a.data}/train.csv", f"{a.data}/val.csv", a.data, a.out, epochs=a.epochs, batch_size=a.batch_size,
              lr=a.lr, num_workers=a.num_workers, seed=a.seed, pretrained=not a.no_pretrained)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
