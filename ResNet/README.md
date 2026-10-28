# ResNet-50 module — fake-image scoring (Fakeddit)

Self-contained image branch of the project. It imports nothing from `fdhn_module` or any other component:
image in, fake score out. Same style as the FDHN module (checkpoint-based `predict()`, documented result,
typed errors, tests).

**Train once, predict forever.** `train.py` runs one time and writes a checkpoint. After that,
`ResNetPredictor(checkpoint).predict(image)` scores any new image with no retraining.

## 1. Install (read the torchvision warning)

```powershell
pip install -r requirements-dev.txt
```

> **torch and torchvision must come from the same build.** You already have a CUDA build of torch for the RTX 4070.
> A plain `pip install torchvision` can pull a *CPU-only* torch over it. Safer, using the same index you used for torch
> (adjust `cu128` if you used a different tag):
> ```powershell
> pip install torchvision pillow pandas --index-url https://download.pytorch.org/whl/cu128
> ```
> then check `python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"`
> prints `True`.

## 2. Workflow

All commands run from inside `resnet_module`. The Fakeddit `.tsv` files are assumed to be in `..\Fakeddit-master`.

```powershell
# 0. offline plumbing check (random weights, synthetic image)
python -m pytest -q
python examples/single_image.py --smoke-test

# 1. PILOT first: 150 images per split, to measure how many Reddit image links still work
python prepare_data.py --data-dir "..\Fakeddit-master" --out-dir pilot --n-train 150 --n-val 50 --n-test 50

# 2. real sample (re-runnable; already-downloaded images are reused)
python prepare_data.py --data-dir "..\Fakeddit-master" --out-dir fakeddit_sample
#    defaults: 6000 train / 1000 val / 1000 test.  Many links are dead: raise --overfetch (e.g. 6) if a split comes up short.

# 3. train once (uses train.csv + val.csv only)
python train.py --data fakeddit_sample --out resnet50_fakeddit.pt

# 4. evaluate ONCE on the unseen test split
python evaluate.py --data fakeddit_sample --weights resnet50_fakeddit.pt

# 5. predict on any new image, no retraining
python examples/single_image.py --weights resnet50_fakeddit.pt --image path\to\photo.jpg
```

**Before step 3, read the "label check" printed at the end of step 2.** It lists the top subreddits per class. The
FAKE list should contain the satire / fake / edited-content subreddits and the REAL list ordinary ones. If it is the
other way round, rerun step 2 with `--real-label 0`. (`--real-label 1` is the default: in the sample row we saw,
`2_way_label=1` came with `6_way_label=0`, which is Fakeddit's "True" class.)

If training shows Windows data-loader errors, add `--num-workers 0`.

## 3. Interface

```python
from resnet_fake import ResNetPredictor
predictor = ResNetPredictor("resnet50_fakeddit.pt")     # load once
r = predictor.predict("photo.jpg")                       # path, bytes or PIL.Image; ONE image per call
```

| `ImagePrediction` field | Meaning |
|---|---|
| `fake_score` | **Float in [0, 1]; 1 = higher predicted likelihood of a fake/misleading image.** `sigmoid(logit)`. |
| `raw_fake_score`, `score_calibrated` | Score before / whether a caller-supplied calibrator was applied. |
| `logit` | Raw model output. |
| `preprocessing` | Original size/mode, source type, low-resolution flag, model input size. |
| `warnings` | Cautions for this call (low resolution, not calibrated, Reddit-trained). |
| `to_dict()` | JSON-safe dict. |

The score is a **ranking score, not a probability**: networks trained with cross-entropy are usually over-confident.
If you fit a calibrator on the *validation* CSV, pass `calibrator=callable` and results are flagged `score_calibrated=True`.

## 4. Data and leakage controls

* The sampler uses the dataset's **official** train / validate / test_public files; rows never move between splits.
* Validation and test rows whose **image URL or normalised title** also appears anywhere in the *full* training split
  are dropped (reposts and cross-posts), and test rows that appear in the full validation split are dropped too.
* In-split duplicates (id, URL, title) are removed.
* `train()` has **no test argument**; the validation split alone picks the best epoch. Only `evaluate.py` reads the test CSV.
* The decision threshold is fixed at 0.5; if you want another, choose it on `--split val`, never on test.
* Class imbalance is handled with a weighted loss; the test set keeps its natural class ratio.
* Tests: `tests/test_sampling.py` (including mutation-checked leak removal), `tests/test_training.py`.

## 5. Errors

All derive from `ResNetFakeError`.

| Situation | Exception |
|---|---|
| image missing / empty / corrupt / truncated / wrong type / < 16 px / > 50 M pixels / > 25 MB | `InvalidImageError` (also a `ValueError`) |
| checkpoint path missing / `None` / not a file | `ModelWeightsNotFoundError` (also a `FileNotFoundError`) |
| corrupt checkpoint, not a state_dict, wrong architecture (e.g. the 1000-class ImageNet head) | `ModelLoadError` |
| forward-pass crash (incl. OOM), NaN/inf output, bad calibrator output | `InferenceError` |
| non-callable calibrator | `InvalidInputError` |

Checkpoints are read with `weights_only=True`, so arbitrary pickles are refused.

## 6. Limitations (put these in your report)

1. **Labels are weak.** Fakeddit labels come from distant supervision (largely the subreddit a post came from), not
   from a human checking each image. The model may learn "looks like a satire/edited-photo subreddit" rather than
   "is factually false". Reported accuracy inherits this noise.
2. **Domain.** Trained on Reddit images; scores on news photos from other sources are out-of-distribution.
3. **Sample size.** Default 6,000 training images, chosen for a fast college project; link rot means the final count
   may be below target (the script says so).
4. **Not measured here.** All tests use random weights or synthetic data; no accuracy figure exists until you train.
5. A single image has no context: the module cannot tell whether a real photo is used in a false caption.
   That is what text scoring and external verification are for.

## 7. Layout

```
resnet_fake/model.py          ResNet-50, single-logit head
resnet_fake/preprocessing.py  image validation + transforms (same eval transform for val/test/predict)
resnet_fake/predictor.py      ResNetPredictor / ImagePrediction
resnet_fake/sampling.py       leak-safe sampling + downloader
resnet_fake/training.py       train() (no test access)     resnet_fake/evaluation.py   evaluate()
resnet_fake/metrics.py        accuracy / precision / recall / F1 / AUC
prepare_data.py  train.py  evaluate.py  examples/single_image.py
tests/                        90 tests
```
