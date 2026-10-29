"""Image files under the data folder (`/data` on Azure, ADR 0003), one images folder per Topic.

Each Revision of an image Content Piece has its own PNG, named from the piece id and the
Revision number, so older Revisions' files are never overwritten. When no image model gives a
usable image, a placeholder card is drawn instead. See spec #21.
"""

import io
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from contentblitz.core.config import get_settings


def image_path(topic_id: str, piece_id: str, revision_number: int) -> Path:
    """Where Revision `revision_number` (from 1) of an image Content Piece is saved."""
    folder = get_settings().contentblitz_data_dir / "images" / topic_id
    return folder / f"{piece_id}-r{revision_number}.png"


def save_png(data: bytes, path: Path) -> tuple[int, int]:
    """Decode the image bytes (PNG, JPEG or WebP), save them as a PNG at `path`, and return
    the image's width and height. Raises PIL's errors for bytes that aren't an image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(data)) as image:
        image.save(path, format="PNG")
        return image.size


PLACEHOLDER_LABEL = "Placeholder image"


def placeholder_png(size: tuple[int, int], colour: str, topic_name: str) -> bytes:
    """A placeholder card as PNG bytes: `colour` ("#RRGGBB") filling `size`, with the Topic
    name and "Placeholder image" centred in a colour that reads on it."""
    background = ImageColor.getrgb(colour)
    luminance = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]
    ink = (20, 20, 20) if luminance > 150 else (245, 245, 245)
    width, height = size
    card = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(card)
    title = ImageFont.load_default(size=max(24, width // 20))
    label = ImageFont.load_default(size=max(16, width // 36))
    draw.text((width / 2, height / 2), topic_name, fill=ink, font=title, anchor="md")
    draw.text((width / 2, height / 2 + 20), PLACEHOLDER_LABEL, fill=ink, font=label, anchor="ma")
    buffer = io.BytesIO()
    card.save(buffer, format="PNG")
    return buffer.getvalue()
