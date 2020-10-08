from dataclasses import dataclass
from functools import lru_cache
from skimage import io
import numpy as np

attr_mapping = {
    "atk": "attack",
    "def": "defense",
    "desc": "description",
    "id": "passcode"
}

@dataclass
class Card:
    passcode: str
    name: str
    type: str
    description: str
    attack: str
    defense: str
    level: str
    race: str
    attribute: str
    scale: str
    archetype: str
    linkval: str
    linkmarkers: str
    image_url: str
    image_url_small: str
    ban_tcg: str
    ban_ocg: str
    ban_goat: str

    def __hash__(self):
        return hash(self.passcode)

    @lru_cache(maxsize=1000)
    def image(self, small:bool=None):
        size = '_small' if small else ''
        image = io.imread(getattr(self, f"image_url{size}"))
        return image