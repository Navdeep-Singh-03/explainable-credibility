# FDHN branch

`inference.py` loads the supplied `BT+TC+CB+FZ.pt` state dict using the matching six-class BERT + TextCNN + CNN-BiLSTM + fuzzy architecture. The original 426 MB file is copied into `checkpoints/`; the SHA-256 matches the Desktop source file.

## Inputs required for inference

FDHN was trained on LIAR statements. For every prediction provide:

- Statement text.
- Six text fields: `subject`, `speaker`, `job_title`, `state_info`, `party_affiliation`, and `context`.
- Five non-negative integer speaker-history counts: `barely_true_counts`, `false_counts`, `half_true_counts`, `mostly_true_counts`, and `pants_on_fire_counts`.

The app collects these in the **FDHN LIAR context** panel. Blank text metadata uses the training preprocessing's `NaN` token. A missing numeric count disables FDHN for that run; it is never guessed.

The predicted class is one of `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`. The app's 0–1 fake score maps the first three classes to fake, `half-true` to half fake, and the last two to real. This is a project-defined ranking score and is not a probability or external truth verification.

## Checkpoint caveat and corrected retraining

The supplied checkpoint was trained from the original script, which passed a two-dimensional `[batch, features]` tensor directly to PyTorch LSTM. PyTorch interprets this as a sequence, so training examples within a batch influence later examples. A single-item inference call is deterministic, but it cannot reproduce the original batch-dependent behavior. The app displays this FDHN output for inspection and excludes this legacy checkpoint from fuzzy fusion.

The training script now fixes the LSTM dimensions and resets metric buffers each epoch. To produce a corrected checkpoint, install dependencies from the project root, download/unzip the LIAR dataset into `SC Proj/liar/` (containing `train.tsv`, `valid.tsv`, and `test.tsv`), then run:

```bash
python fdhn/model/2.2.BT+TC+CB+FZ.py
```

It saves `fdhn/checkpoints/BT+TC+CB+FZ-corrected.pt` without overwriting the supplied checkpoint. The app can use that file after setting:

```bash
export SC_PROJ_FDHN_CHECKPOINT="fdhn/checkpoints/BT+TC+CB+FZ-corrected.pt"
```

Windows PowerShell:

```powershell
$env:SC_PROJ_FDHN_CHECKPOINT = "fdhn/checkpoints/BT+TC+CB+FZ-corrected.pt"
```

Only a checkpoint whose metadata contains `lstm_batch_order_bug_fixed: true` is eligible to influence fuzzy fusion. The trainer writes this marker into its corrected checkpoint; renaming a legacy checkpoint does not make it eligible. Training takes time and uses the LIAR dataset and BERT tokenizer/model configuration. The supplied legacy checkpoint's accompanying log reports 42.38% test accuracy and 0.4253 macro F1; those metrics are recorded outputs, not independently reproduced here.

## Upstream

The upstream implementation describes the tripartite text/context/numeric architecture and is MIT-licensed. The license is retained in `LICENSE`.

- Repository: https://github.com/chengxuphd/FDHN
- Paper: https://doi.org/10.1145/3628797.3628971
