"""One-off training of the ResNet-50 classifier. Run once; predict forever without retraining.

Leakage design: ``train()`` accepts ONLY a train CSV and a validation CSV. The validation set
chooses the best epoch; the test set is deliberately not an argument here and is read only by
``evaluation.evaluate`` after training is finished.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ImageCsvDataset
from .metrics import classification_report
from .model import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, build_model
from .preprocessing import eval_transform, train_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _validate(model, loader, device) -> Dict[str, Any]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).squeeze(1).float().cpu())
        labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    loss = F.binary_cross_entropy_with_logits(logits, labels).item()
    scores = torch.sigmoid(logits).tolist()
    report = classification_report([int(v) for v in labels.tolist()], scores)
    report["loss"] = loss
    return report


def train(
    train_csv: str,
    val_csv: str,
    root: str,
    out_path: str,
    *,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: Optional[Union[str, torch.device]] = None,
    pretrained: bool = True,
    num_workers: int = 2,
    seed: int = 0,
    image_size: int = IMAGE_SIZE,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Fine-tune ResNet-50 and save the best-on-validation checkpoint to ``out_path``."""
    set_seed(seed)
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train = ImageCsvDataset(train_csv, root, train_transform(image_size))
    ds_val = ImageCsvDataset(val_csv, root, eval_transform(image_size))
    n_fake = sum(ds_train.labels)
    n_real = len(ds_train) - n_fake
    if n_fake == 0 or n_real == 0:
        raise ValueError("training split contains only one class")
    log(f"device={device} train={len(ds_train)} (fake {n_fake} / real {n_real}) val={len(ds_val)}")

    drop_last = len(ds_train) > batch_size  # BatchNorm cannot train on a batch of 1
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last)
    val_loader = DataLoader(ds_val, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)

    model = build_model(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = torch.tensor(n_real / n_fake, device=device)  # compensates class imbalance
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: List[Dict[str, Any]] = []
    best_key, best = -math.inf, None
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logit = model(x).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_weight)
            if not math.isfinite(loss.item()):
                raise RuntimeError("training diverged (non-finite loss); try a smaller --lr")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(x)
            seen += len(x)
        val = _validate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val["loss"],
               "val_accuracy": val["accuracy"], "val_f1_fake": val["f1_fake"], "val_auc": val["roc_auc"]}
        history.append(row)
        log(f"epoch {epoch}/{epochs}  train_loss={row['train_loss']:.4f}  val_loss={row['val_loss']:.4f}  "
            f"val_acc={row['val_accuracy']:.4f}  val_f1={row['val_f1_fake']:.4f}  val_auc={row['val_auc']:.4f}")

        key = val["roc_auc"] if math.isfinite(val["roc_auc"]) else -val["loss"]  # select on VALIDATION only
        if key > best_key:
            best_key, best = key, row
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "meta": {
                        "architecture": "resnet50-single-logit", "image_size": image_size,
                        "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
                        "score_meaning": "sigmoid(logit) = fake score, 1 = fake",
                        "n_train": len(ds_train), "n_val": len(ds_val),
                        "train_fake": n_fake, "train_real": n_real,
                        "best_epoch": epoch, "val_auc": row["val_auc"], "val_accuracy": row["val_accuracy"],
                        "init": "ImageNet IMAGENET1K_V2" if pretrained else "random",
                    },
                },
                out_path,
            )
            log(f"  saved best checkpoint -> {out_path}")
    return {"history": history, "best": best, "checkpoint": out_path}
