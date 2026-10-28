#!/usr/bin/env python3
"""Score one image with a trained checkpoint (no retraining).

    python examples/single_image.py --weights resnet50_fakeddit.pt --image photo.jpg
    python examples/single_image.py --smoke-test        # offline plumbing check, RANDOM weights
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resnet_fake import ResNetFakeError, ResNetPredictor  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights")
    p.add_argument("--image")
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.smoke_test:
            import torch
            from PIL import Image
            from resnet_fake.model import build_model

            print("*** SMOKE TEST: random weights and a synthetic image. The score is NOT a real prediction. ***\n", file=sys.stderr)
            with tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(0)
                ckpt = Path(tmp) / "random.pt"
                torch.save(build_model(False).state_dict(), ckpt)
                img = Path(tmp) / "synthetic.png"
                Image.effect_noise((256, 256), 60).convert("RGB").save(img)
                result = ResNetPredictor(ckpt, device="cpu").predict(img)
        else:
            if not a.image:
                p.error("--image is required unless --smoke-test is used")
            result = ResNetPredictor(a.weights).predict(a.image)
    except ResNetFakeError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
