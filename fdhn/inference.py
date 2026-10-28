"""Inference adapter for the upstream FDHN BT+TC+CB+FZ LIAR checkpoint.

Architecture and tensor dimensions follow ``model/2.2.BT+TC+CB+FZ.py``. Imports of torch
and transformers are delayed so the rest of the app remains usable when those optional
dependencies are not installed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

FDHN_LABELS = ("pants-fire", "false", "barely-true", "half-true", "mostly-true", "true")
TEXT_CONTEXT_FIELDS = ("subject", "speaker", "job_title", "state_info", "party_affiliation", "context")
NUMERIC_CONTEXT_FIELDS = ("barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts")


class FDHNUnavailable(RuntimeError):
    """The checkpoint, model dependencies, or required LIAR metadata are unavailable."""


def fake_score_from_probabilities(probabilities: Sequence[float]) -> float:
    """Map LIAR's 6 labels to a 0..1 low-truthiness ranking score.

    `pants-fire`, `false`, and `barely-true` contribute fully; `half-true` contributes
    half; `mostly-true` and `true` contribute zero. This is a project mapping, not an
    output/calibrated probability defined by the original FDHN paper.
    """
    if len(probabilities) != 6:
        raise ValueError("FDHN must return six class scores")
    values = [float(x) for x in probabilities]
    if any(x < 0 or x > 1 for x in values):
        raise ValueError("class scores must be in [0, 1]")
    return sum(values[:3]) + 0.5 * values[3]


def split_checkpoint(checkpoint_object):
    """Accept a raw upstream state dict or our metadata-wrapped corrected checkpoint."""
    if not isinstance(checkpoint_object, Mapping) or not checkpoint_object:
        raise ValueError("checkpoint is not a non-empty mapping")
    if "state_dict" in checkpoint_object:
        state = checkpoint_object["state_dict"]
        metadata = checkpoint_object.get("meta", {})
        if not isinstance(state, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError("checkpoint state_dict/meta has an invalid format")
        return state, metadata
    return checkpoint_object, {}


def validate_inputs(text: str, text_context: Mapping[str, str], numeric_context: Mapping[str, object]) -> tuple[dict[str, str], list[int]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("FDHN statement text must not be empty")
    if not isinstance(text_context, Mapping):
        raise ValueError("text_context must be a mapping of LIAR metadata fields")
    if not isinstance(numeric_context, Mapping):
        raise ValueError("numeric_context must be a mapping of five LIAR history counts")
    text_values = {}
    for name in TEXT_CONTEXT_FIELDS:
        value = text_context.get(name, "NaN")
        if value is None or not str(value).strip():
            value = "NaN"  # matches the original LIAR training preprocessing for missing text metadata
        text_values[name] = str(value)
    counts = []
    for name in NUMERIC_CONTEXT_FIELDS:
        value = numeric_context.get(name)
        if value is None or isinstance(value, bool):
            raise ValueError(f"{name} is required; enter a known non-negative LIAR history count")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a non-negative integer") from exc
        if str(value).strip() not in (str(number), f"{number}.0") or number < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        counts.append(number)
    return text_values, counts


def _build_network(torch, nn, BertConfig, BertForSequenceClassification):
    class FuzzyLayer(nn.Module):
        def __init__(self, input_dim, membership_num):
            super().__init__()
            self.input_dim = input_dim
            self.membership_num = membership_num
            self.membership_miu = nn.Parameter(torch.Tensor(membership_num, input_dim), requires_grad=True)
            self.membership_sigma = nn.Parameter(torch.Tensor(membership_num, input_dim), requires_grad=True)
            nn.init.xavier_uniform_(self.membership_miu)
            nn.init.ones_(self.membership_sigma)

        def forward(self, input_seq):
            b = input_seq.size(0)
            x = input_seq.unsqueeze(1).expand(b, self.membership_num, self.input_dim)
            mu = self.membership_miu.unsqueeze(0).expand(b, self.membership_num, self.input_dim)
            sigma = self.membership_sigma.unsqueeze(0).expand(b, self.membership_num, self.input_dim)
            return torch.mean(torch.exp(-0.5 * ((x - mu) / sigma) ** 2), dim=-1)

    class TextCNN(nn.Module):
        def __init__(self, vocab_size=30522, embedding_dim=60, n_filters=128,
                     filter_sizes=(3, 4, 5), output_dim=6, dropout=0.5, pad_idx=0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, n_filters, kernel_size=size) for size in filter_sizes])
            self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, text):
            embedded = self.embedding(text).permute(0, 2, 1)
            conved = [torch.relu(conv(embedded)) for conv in self.convs]
            pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
            return self.fc(self.dropout(torch.cat(pooled, dim=1)))

    class CNNBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(5, 128)
            self.conv = nn.Conv1d(128, 32, kernel_size=1)
            self.rnn = nn.LSTM(32, 64, num_layers=2, bidirectional=True, dropout=0.5)
            self.fc = nn.Linear(128, 6)
            self.dropout = nn.Dropout(0.5)

        def forward(self, metadata):
            embedded = self.dropout(self.embedding(metadata))
            embedded = torch.reshape(embedded, (metadata.size(0), 128, 1))
            conved = torch.relu(self.conv(embedded))
            conved = torch.reshape(conved, (metadata.size(0), 32))
            # Corrected layout: each record is a one-step sequence in the batch dimension.
            outputs, _ = self.rnn(conved.unsqueeze(0))
            return self.fc(outputs[-1])

    class LiarModel(nn.Module):
        def __init__(self):
            super().__init__()
            bert_config = BertConfig(
                vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                num_attention_heads=12, intermediate_size=3072,
                max_position_embeddings=512, type_vocab_size=2, num_labels=6,
                pad_token_id=0, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                layer_norm_eps=1e-12, initializer_range=0.02,
            )
            self.bert = BertForSequenceClassification(bert_config)
            self.textcnn2 = TextCNN()
            self.cnn_bilstm = CNNBiLSTM()
            self.fuzzy = FuzzyLayer(6, 6)
            self.fuse = nn.Linear(24, 6)

        def forward(self, text, text_masks, metadata_text, metadata_number):
            text_output = self.bert(input_ids=text, attention_mask=text_masks).logits
            metadata_output_text = self.textcnn2(metadata_text)
            metadata_output_number = self.cnn_bilstm(metadata_number)
            metadata_output_fuzzy = self.fuzzy(metadata_output_number)
            return self.fuse(torch.cat((text_output, metadata_output_text,
                                        metadata_output_number, metadata_output_fuzzy), dim=1))

    return LiarModel()


class FDHNPredictor:
    """Load the supplied FDHN state dict and score one LIAR-style statement at a time."""

    def __init__(self, checkpoint: str | os.PathLike, *, device: str | None = None,
                 tokenizer_name: str = "bert-base-uncased", local_files_only: bool = False):
        path = Path(checkpoint)
        if not path.is_file():
            raise FDHNUnavailable(f"FDHN checkpoint not found: {path}")
        try:
            import torch
            from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        except ImportError as exc:
            raise FDHNUnavailable("FDHN inference requires torch and transformers; install requirements.txt") from exc
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
            self.model = _build_network(torch, torch.nn, BertConfig, BertForSequenceClassification)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            state, metadata = split_checkpoint(loaded)
            if not isinstance(state, Mapping) or not state:
                raise ValueError("checkpoint is not a non-empty state_dict")
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device).eval()
            self.corrected_training = metadata.get("lstm_batch_order_bug_fixed") is True
        except Exception as exc:
            raise FDHNUnavailable(f"Could not load FDHN checkpoint/tokenizer ({type(exc).__name__}: {exc})") from exc
        self.checkpoint = str(path)

    def predict(self, text: str, *, text_context: Mapping[str, str],
                numeric_context: Mapping[str, object]) -> dict:
        text_values, counts = validate_inputs(text, text_context, numeric_context)
        t = self.torch
        try:
            statement = self.tokenizer(text, truncation=True, padding="max_length", max_length=128,
                                       return_tensors="pt")
            metadata_tokens = []
            for name in TEXT_CONTEXT_FIELDS:
                encoded = self.tokenizer(text_values[name], truncation=True, padding="max_length",
                                         max_length=10, return_tensors="pt")
                metadata_tokens.append(encoded["input_ids"])
            metadata_text = t.cat(metadata_tokens, dim=1).to(self.device)
            metadata_number = t.tensor([counts], dtype=t.float32, device=self.device)
            with t.inference_mode():
                logits = self.model(statement["input_ids"].to(self.device),
                                    statement["attention_mask"].to(self.device),
                                    metadata_text, metadata_number)[0]
                # Training used BCEWithLogitsLoss on one-hot targets and argmax. Softmax is
                # used here only to form a normalized ranking distribution, not calibrated odds.
                class_scores = t.softmax(logits.float(), dim=-1).cpu().tolist()
            index = max(range(6), key=class_scores.__getitem__)
            warnings = ["trained on LIAR political statements and speaker metadata; general news articles are out of domain"]
            if not self.corrected_training:
                warnings.append("this is a legacy checkpoint trained with the original 2D [batch, features] LSTM call; batch rows were treated as sequence steps, making predictions batch-order dependent")
            return {
                "model": "FDHN BT+TC+CB+FZ on LIAR",
                "checkpoint": self.checkpoint,
                "predicted_class": FDHN_LABELS[index],
                "class_scores": {name: float(score) for name, score in zip(FDHN_LABELS, class_scores)},
                "fake_score": fake_score_from_probabilities(class_scores),
                "score_note": "LIAR truthfulness ranking mapped from six classes; not a calibrated probability or external verification.",
                "metadata_inputs_used": {**text_values, **dict(zip(NUMERIC_CONTEXT_FIELDS, counts))},
                "fusion_eligible": self.corrected_training,
                "warnings": warnings,
            }
        except Exception as exc:
            raise FDHNUnavailable(f"FDHN inference failed ({type(exc).__name__}: {exc})") from exc
