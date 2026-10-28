"""Leak-safe sampling and image download for Fakeddit.

Why this exists: the official ``image_downloader.py`` tries to fetch EVERY row (564k train
rows alone) with no timeout or error handling. We only need a few thousand images, so this
module draws a stratified sample from each OFFICIAL split and downloads just those.

Leakage controls (all enforced here and unit-tested):
* Splits come from the dataset's own train / validate / test_public files; rows never move
  between splits.
* Any validation/test row whose image URL OR normalised title also occurs anywhere in the
  full training split is dropped (reposts and cross-posts would otherwise leak). Test rows
  that also occur in the full validation split are dropped as well.
* Duplicate ids / image URLs / titles inside a split are dropped.
* The sampled test set is only ever read by ``evaluate.py``; ``train.py`` has no argument for it.

Label convention: Fakeddit ``2_way_label`` marks the REAL class with ``real_label`` (default 1);
everything else becomes ``is_fake = 1``. The CLI prints a per-class subreddit table so you can
confirm this on your own data before training.
"""

from __future__ import annotations

import os
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from .exceptions import InvalidImageError
from .preprocessing import load_image

REQUIRED_COLS = ["id", "clean_title", "image_url", "hasImage", "2_way_label", "subreddit"]
LABEL_COL = "2_way_label"
USER_AGENT = "Mozilla/5.0 (compatible; college-project-fake-news-research/1.0)"
OUT_COLS = ["id", "clean_title", "image_path", "is_fake", "subreddit"]


# --------------------------------------------------------------------------- reading
def read_split(path: str) -> pd.DataFrame:
    """Read the needed columns of an official Fakeddit TSV."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fakeddit file not found: {path}")
    try:
        return pd.read_csv(path, sep="\t", usecols=REQUIRED_COLS, dtype=str)
    except ValueError as exc:
        header = list(pd.read_csv(path, sep="\t", nrows=0).columns)
        raise ValueError(f"{path} lacks one of the columns {REQUIRED_COLS}; found {header}") from exc


def _title_key(series: pd.Series) -> pd.Series:
    return series.fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def key_sets(df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """(image urls, normalised titles) of a RAW split, used to detect cross-split duplicates."""
    urls = set(df["image_url"].dropna().astype(str)) - {"", "nan"}
    titles = set(_title_key(df["clean_title"])) - {""}
    return urls, titles


# ------------------------------------------------------------------------- cleaning
def clean_candidates(df: pd.DataFrame, real_label: int = 1) -> pd.DataFrame:
    """Keep rows with a usable image URL, title and 0/1 label; add ``is_fake``; drop in-split duplicates."""
    out = df.copy()
    has_image = out["hasImage"].astype(str).str.lower().eq("true")
    url_ok = out["image_url"].fillna("").str.startswith(("http://", "https://"))
    label = pd.to_numeric(out[LABEL_COL], errors="coerce")
    out["title_key"] = _title_key(out["clean_title"])
    out = out[has_image & url_ok & label.isin([0, 1]) & out["title_key"].ne("")].copy()
    out["is_fake"] = (pd.to_numeric(out[LABEL_COL]) != real_label).astype(int)
    out = out.drop_duplicates("id").drop_duplicates("image_url").drop_duplicates("title_key")
    return out[["id", "clean_title", "image_url", "subreddit", "is_fake", "title_key"]].reset_index(drop=True)


def remove_leaks(df: pd.DataFrame, urls: Set[str], titles: Set[str]) -> pd.DataFrame:
    """Drop rows whose image URL or normalised title appears in an earlier split."""
    keep = ~(df["image_url"].isin(urls) | df["title_key"].isin(titles))
    return df[keep].reset_index(drop=True)


def stratified_candidates(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Shuffled sample of ``n`` rows keeping the split's natural class proportions."""
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled = df.groupby("is_fake", group_keys=False).sample(frac=n / len(df), random_state=seed)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_candidates(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    overfetch: float = 3.0,
    seed: int = 42,
    real_label: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Ordered candidate rows per split (``overfetch`` x the target, since many links are dead)."""
    train_urls, train_titles = key_sets(train_raw)
    val_urls, val_titles = key_sets(val_raw)

    tr = clean_candidates(train_raw, real_label)
    va = remove_leaks(clean_candidates(val_raw, real_label), train_urls, train_titles)
    te = remove_leaks(clean_candidates(test_raw, real_label), train_urls | val_urls, train_titles | val_titles)

    def take(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return stratified_candidates(df, int(n * overfetch), seed)

    return {"train": take(tr, n_train), "val": take(va, n_val), "test": take(te, n_test)}


# ------------------------------------------------------------------------ downloading
def download_image(url: str, dest: str, *, timeout: float = 10.0, max_bytes: int = 10 * 1024 * 1024,
                   max_side: int = 256) -> Tuple[bool, str]:
    """Download ``url``, validate it decodes as an image, store it as a <=256px JPEG. Never raises."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
    except Exception as exc:  # dead link, HTTP error, timeout, DNS ...
        return False, f"download_failed:{type(exc).__name__}"
    if len(data) > max_bytes:
        return False, "too_large"
    try:
        img, _ = load_image(data)
        img.thumbnail((max_side, max_side))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp"
        img.save(tmp, format="JPEG", quality=90)
        os.replace(tmp, dest)  # atomic: never leaves a half-written file
    except (InvalidImageError, OSError) as exc:
        return False, f"not_an_image:{type(exc).__name__}"
    return True, "ok"


def _safe_name(raw_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(raw_id))


def _fetch_one(rec: Dict[str, Any], img_dir: str, downloader: Callable, timeout: float) -> Tuple[bool, str]:
    dest = os.path.join(img_dir, _safe_name(rec["id"]) + ".jpg")
    if os.path.isfile(dest):  # resume support
        try:
            load_image(dest)
            return True, "cached"
        except InvalidImageError:
            os.remove(dest)
    return downloader(rec["image_url"], dest, timeout=timeout)


def fetch_split(
    cands: pd.DataFrame,
    out_dir: str,
    split: str,
    target: int,
    *,
    workers: int = 16,
    timeout: float = 10.0,
    downloader: Callable = download_image,
    log: Callable[[str], None] = print,
) -> Tuple[pd.DataFrame, Counter]:
    """Walk the candidates in order and keep the first ``target`` that download successfully."""
    img_dir = os.path.join(out_dir, "images", split)
    records = cands.to_dict("records")
    kept: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    chunk = max(workers * 4, 32)
    pos = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(kept) < target and pos < len(records):
            batch = records[pos:pos + chunk]
            pos += chunk
            futures = [pool.submit(_fetch_one, r, img_dir, downloader, timeout) for r in batch]
            for rec, fut in zip(batch, futures):
                ok, reason = fut.result()
                stats[reason.split(":")[0]] += 1
                if ok and len(kept) < target:
                    kept.append({
                        "id": rec["id"], "clean_title": rec["clean_title"],
                        "image_path": f"images/{split}/{_safe_name(rec['id'])}.jpg",
                        "is_fake": int(rec["is_fake"]), "subreddit": rec["subreddit"],
                    })
            log(f"[{split}] {len(kept)}/{target} images ({pos}/{len(records)} candidates tried)")
    if len(kept) < target:
        log(f"[{split}] WARNING: only {len(kept)}/{target} images could be downloaded (ran out of candidates)")
    return pd.DataFrame(kept, columns=OUT_COLS), stats


# ------------------------------------------------------------------------ orchestration
def label_sanity_table(df: pd.DataFrame, top: int = 8) -> str:
    """Top subreddits per class: lets you confirm which Fakeddit label value means 'fake'."""
    lines = []
    for label, name in ((1, "FAKE (is_fake=1)"), (0, "REAL (is_fake=0)")):
        sub = df[df["is_fake"] == label]["subreddit"].value_counts().head(top)
        lines.append(f"  {name}: " + ", ".join(f"{s} ({c})" for s, c in sub.items()))
    return "\n".join(lines)


def prepare(
    data_dir: str,
    out_dir: str,
    *,
    n_train: int = 6000,
    n_val: int = 1000,
    n_test: int = 1000,
    overfetch: float = 3.0,
    seed: int = 42,
    real_label: int = 1,
    workers: int = 16,
    timeout: float = 10.0,
    files: Tuple[str, str, str] = ("multimodal_train.tsv", "multimodal_validate.tsv", "multimodal_test_public.tsv"),
    downloader: Callable = download_image,
    log: Callable[[str], None] = print,
) -> Dict[str, pd.DataFrame]:
    """Sample, de-leak and download; writes ``train.csv``, ``val.csv``, ``test.csv`` into ``out_dir``."""
    log("reading official splits ...")
    train_raw, val_raw, test_raw = (read_split(os.path.join(data_dir, f)) for f in files)
    cands = build_candidates(train_raw, val_raw, test_raw, n_train=n_train, n_val=n_val, n_test=n_test,
                             overfetch=overfetch, seed=seed, real_label=real_label)
    os.makedirs(out_dir, exist_ok=True)
    result: Dict[str, pd.DataFrame] = {}
    for split, target in (("train", n_train), ("val", n_val), ("test", n_test)):
        df, stats = fetch_split(cands[split], out_dir, split, target, workers=workers, timeout=timeout,
                                downloader=downloader, log=log)
        df.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
        result[split] = df
        log(f"[{split}] saved {len(df)} rows; outcomes: {dict(stats)}; fake share: "
            f"{df['is_fake'].mean():.2%}" if len(df) else f"[{split}] saved 0 rows")

    # Final guard: the three written splits must be disjoint.
    ids = [set(result[s]["id"]) for s in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "split overlap detected"
    if len(result["train"]):
        log("label check (does FAKE list the satire/edited/fake subreddits you expect?):\n" + label_sanity_table(result["train"]))
    return result
