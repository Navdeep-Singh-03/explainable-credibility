#!/usr/bin/env python3
"""Step 1: sample the official Fakeddit splits and download ONLY the images we need.

    python prepare_data.py --data-dir "../Fakeddit-master" --out-dir fakeddit_sample

Safe to re-run: images already downloaded are reused. Writes train.csv / val.csv / test.csv.
"""
import argparse
import sys

from resnet_fake.sampling import prepare


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder containing the three multimodal_*.tsv files")
    p.add_argument("--out-dir", default="fakeddit_sample")
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--overfetch", type=float, default=3.0, help="candidates tried per wanted image (dead links are common)")
    p.add_argument("--real-label", type=int, default=1, choices=[0, 1], help="which 2_way_label value means REAL")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    try:
        prepare(a.data_dir, a.out_dir, n_train=a.n_train, n_val=a.n_val, n_test=a.n_test, overfetch=a.overfetch,
                seed=a.seed, real_label=a.real_label, workers=a.workers, timeout=a.timeout)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
