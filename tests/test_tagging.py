"""Verify tag ranking and filtering without downloading a model."""
import torch
from PIL import Image

from upscaler.tagging import SPORTS_LABELS, SportsTagger


def test_tags_are_sorted_and_low_scores_are_removed():
    class Processor:
        def __call__(self, **kwargs):
            return {"pixel_values": torch.ones(1, 3, 2, 2)}

    class Model:
        def get_image_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0]])

    tagger = SportsTagger.__new__(SportsTagger)
    tagger.device = torch.device("cpu")
    tagger.processor = Processor()
    tagger.model = Model()
    tagger.text_features = torch.tensor([[0.30, 0.0], [0.80, 0.0]] +
                                         [[0.1, 0.0]] * (len(SPORTS_LABELS) - 2))
    assert tagger.tag(Image.new("RGB", (2, 2))) == [
        {"label": SPORTS_LABELS[1][0], "score": 0.8},
        {"label": SPORTS_LABELS[0][0], "score": 0.3},
    ]
