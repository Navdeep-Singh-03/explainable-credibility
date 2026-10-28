"""Image loading, validation and tensor transforms.

The SAME ``eval_transform`` is used for validation, test and prediction, so what the
model is evaluated on is exactly what it sees at prediction time.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms as T

from .exceptions import InvalidImageError
from .model import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 50_000_000
MIN_SIDE = 16  # below this the image is rejected
LOW_RES_SIDE = 64  # below this a warning is attached


def eval_transform(image_size: int = IMAGE_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> T.Compose:
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)])


def train_transform(image_size: int = IMAGE_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def _to_rgb(img: Image.Image) -> Image.Image:
    """Flatten transparency onto white, then convert to 3-channel RGB."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def load_image(image: Any) -> Tuple[Image.Image, Dict[str, Any]]:
    """Return ``(RGB PIL image, info)`` from a path, raw bytes or a PIL image.

    Raises :class:`InvalidImageError` for anything unusable.
    """
    source: str
    try:
        if isinstance(image, Image.Image):
            source = "pil"
            img = image.copy()
        elif isinstance(image, (bytes, bytearray)):
            source = "bytes"
            if len(image) == 0:
                raise InvalidImageError("image bytes are empty")
            if len(image) > MAX_FILE_BYTES:
                raise InvalidImageError(f"image is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
            img = Image.open(io.BytesIO(bytes(image)))
        elif isinstance(image, (str, os.PathLike)):
            source = "path"
            path = os.fspath(image)
            if not path:
                raise InvalidImageError("image path is empty")
            if not os.path.isfile(path):
                raise InvalidImageError(f"image file not found: {path!r}")
            size = os.path.getsize(path)
            if size == 0:
                raise InvalidImageError(f"image file is empty: {path!r}")
            if size > MAX_FILE_BYTES:
                raise InvalidImageError(f"image is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB: {path!r}")
            img = Image.open(path)
        else:
            raise InvalidImageError(f"image must be a path, bytes or PIL.Image, got {type(image).__name__}")

        width, height = img.size  # header only for lazily opened files
        if width * height > MAX_PIXELS:
            raise InvalidImageError(f"image has {width * height:,} pixels (limit {MAX_PIXELS:,})")
        img.load()  # decode now so truncated/corrupt files fail here
        img = ImageOps.exif_transpose(img)
        original_mode = img.mode
        img = _to_rgb(img)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError, EOFError) as exc:
        raise InvalidImageError(f"cannot decode image ({type(exc).__name__}: {exc})") from exc

    width, height = img.size
    if min(width, height) < MIN_SIDE:
        raise InvalidImageError(f"image is too small ({width}x{height}); minimum side is {MIN_SIDE}px")
    info = {"source": source, "original_size": [width, height], "original_mode": original_mode,
            "low_resolution": min(width, height) < LOW_RES_SIDE}
    return img, info
