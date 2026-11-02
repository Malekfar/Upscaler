"""Local zero-shot sports and recreation image tagging using CLIP."""
from pathlib import Path
import os

import torch

from upscaler.engine import resolve_device

# Descriptions are English because the original CLIP model is English-trained;
# returned labels are Persian. Scores are cosine similarities, not probabilities.
SPORTS_LABELS = (
    ("استخر", "a swimming pool"),
    ("جکوزی", "a jacuzzi hot tub with bubbling water"),
    ("سونا", "a sauna room"),
    ("سرسره آبی", "a water slide"),
    ("پارک آبی", "a water park"),
    ("استخر سرپوشیده", "an indoor swimming pool"),
    ("استخر روباز", "an outdoor swimming pool"),
    ("باشگاه بدنسازی", "a fitness gym with exercise equipment"),
    ("وزنه", "dumbbells and weightlifting equipment"),
    ("تردمیل", "a treadmill exercise machine"),
    ("سالن ورزشی", "an indoor sports hall"),
    ("زمین فوتبال", "a soccer football field"),
    ("زمین تنیس", "a tennis court"),
    ("زمین بسکتبال", "a basketball court"),
    ("زمین والیبال", "a volleyball court"),
    ("دوچرخه ثابت", "a stationary exercise bike"),
    ("پیست دو", "a running track"),
    ("مجموعه ورزشی", "a sports complex"),
)
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DEFAULT_THRESHOLD = 0.26


class SportsTagger:
    """Load CLIP once and reuse text embeddings for all queued images."""

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor

        self.device = resolve_device("auto")
        weights_dir = os.environ.get("UPSCALER_WEIGHTS_DIR")
        cache_dir = str(Path(weights_dir) / "huggingface") if weights_dir else None
        options = {"revision": MODEL_REVISION, "cache_dir": cache_dir}
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False, **options)
        self.model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=False, **options).to(self.device).eval()
        inputs = self.processor(text=["a photo of " + text for _, text in SPORTS_LABELS],
                                padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
            self.text_features = features / features.norm(dim=-1, keepdim=True)

    def tag(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            scores = (features @ self.text_features.T)[0].cpu().tolist()
        return [{"label": label, "score": round(score, 4)}
                for (label, _), score in sorted(zip(SPORTS_LABELS, scores),
                                                 key=lambda item: item[1], reverse=True)
                if score >= DEFAULT_THRESHOLD]
