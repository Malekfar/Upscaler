"""Local zero-shot sports and recreation image tagging using CLIP."""
from pathlib import Path
import os

import torch

from upscaler.engine import resolve_device

# Descriptions are English because the original CLIP model is English-trained;
# returned labels are Persian. Scores are cosine similarities, not probabilities.
SPORTS_LABELS = (
    ("استخر", "a swimming pool"),
    ("اسپا و ماساژ", "a spa massage treatment"),
    ("حمام ترکی", "a Turkish bath hammam"),
    ("کلاس شنا", "a swimming lesson with an instructor and students"),
    ("پکیج شنا", "a swimming pool and swimming lesson"),
    ("مرکز تفریحی", "a recreation center with leisure facilities"),
    ("پارک آبی", "a water park with pools and water slides"),
    ("باشگاه بدنسازی", "a fitness gym with weight training equipment"),
    ("لوازم ورزشی", "sports equipment and athletic gear"),
    ("استخر روباز", "an outdoor swimming pool"),
    ("باشگاه رزمی", "a martial arts gym with people training"),
    ("ژیمناستیک", "gymnasts practicing gymnastics in a gymnasium"),
    ("تکواندو", "taekwondo martial artists practicing kicks"),
    ("کاراته", "karate martial artists practicing in a dojo"),
    ("کنگ فو", "kung fu martial artists practicing"),
    ("بوکس", "boxers wearing boxing gloves in a boxing ring"),
    ("کیک بوکسینگ", "kickboxing fighters training with gloves"),
    ("قایقرانی", "people paddling a boat on the water"),
    ("پدل برد", "a person standing on a paddleboard on water"),
    ("سالن چند منظوره", "an indoor multipurpose sports hall"),
    ("سالن والیبال", "an indoor volleyball court with a net"),
    ("سالن فوتسال", "an indoor futsal soccer court"),
    ("سالن پینگ پنگ", "an indoor table tennis hall with ping pong tables"),
    ("سالن بسکتبال", "an indoor basketball court with hoops"),
    ("زمین چمن", "a natural grass sports field"),
    ("سالن هندبال", "an indoor handball court with goals"),
    ("تنیس", "people playing tennis on a tennis court"),
    ("چمن مصنوعی", "an artificial turf sports field"),
    ("باشگاه سوارکاری", "an equestrian riding club with horses"),
    ("باشگاه تیراندازی", "an indoor shooting range with shooting targets"),
    ("فیتنس", "people doing fitness training in a gym"),
    ("یوگا", "people practicing yoga on mats"),
    ("ایروبیک", "a group doing aerobics exercise"),
    ("پیلاتس", "people doing pilates exercises"),
    ("کراس فیت", "athletes doing crossfit training in a gym"),
    ("تی آر ایکس", "a person exercising with TRX suspension straps"),
    ("سوارکاری", "a person riding a horse"),
    ("تیراندازی", "a person practicing target shooting at a range"),
    ("شهربازی", "an amusement park with rides"),
    ("بولینگ", "people playing bowling at a bowling alley"),
    ("تم پارک", "a themed amusement park with attractions"),
    ("خانه بازی", "an indoor children's play center with toys"),
    ("پینت بال", "people playing paintball wearing protective masks"),
    ("بیلیارد و اسنوکر", "a billiards or snooker table with balls and cues"),
    ("اتاق فرار", "a themed escape room game with puzzles"),
    ("آبدرمانی", "a hydrotherapy pool for water therapy"),
    ("مرکز لیزر", "a laser hair removal treatment at a beauty clinic"),
    ("اسکیت", "a person roller skating"),
    ("اسکیت برد", "a person riding a skateboard"),
    ("اسکیت روی یخ", "a person ice skating on an ice rink"),
    ("فیشیال", "a facial skin care treatment at a beauty salon"),
    ("سالن کشتی", "a wrestling training hall with wrestling mats"),
    ("بدمینتون", "people playing badminton with rackets and a shuttlecock"),
    ("پاراگلایدرسواری", "a person paragliding in the sky"),
    ("کلاس فوتبال", "a football coach teaching children on a soccer field"),
    ("پدل", "people playing padel tennis on an enclosed court"),
    ("تلکابین", "a gondola cable car in the mountains"),
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
