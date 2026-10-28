"""Optional text/image semantic similarity. Requires transformers + torch + downloaded CLIP weights."""
from __future__ import annotations


class SimilarityUnavailable(RuntimeError):
    pass


class ClipSimilarity:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise SimilarityUnavailable("Install the optional CLIP dependencies in requirements-clip.txt") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        except Exception as exc:
            raise SimilarityUnavailable(f"Could not load CLIP model {model_name!r}: {exc}") from exc

    def score(self, text: str, image) -> dict:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        try:
            inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.inference_mode():
                t = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
                i = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                t = self.torch.nn.functional.normalize(t, dim=-1)
                i = self.torch.nn.functional.normalize(i, dim=-1)
                cosine = float((t * i).sum(dim=-1).item())
            # Convert similarity to a relative mismatch signal; this mapping is heuristic, not truth evidence.
            mismatch = max(0.0, min(1.0, (0.30 - cosine) / 0.60))
            return {"cosine_similarity": cosine, "inconsistency": mismatch,
                    "note": "CLIP semantic consistency is not evidence that a claim is true."}
        except Exception as exc:
            raise SimilarityUnavailable(f"CLIP inference failed: {exc}") from exc
