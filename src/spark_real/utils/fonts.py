"""Monospace PIL font for image annotation, with the PIL default as fallback."""

from PIL import ImageFont

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def load_font(size: int = 14):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except (OSError, IOError):
        return ImageFont.load_default()
