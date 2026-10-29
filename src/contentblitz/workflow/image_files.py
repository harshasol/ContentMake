"""Image files under the data folder (`/data` on Azure, ADR 0003), one images folder per Topic.

Each Revision of an image Content Piece has its own PNG, named from the piece id and the
Revision number, so older Revisions' files are never overwritten. See spec #21.
"""

import io
from pathlib import Path

from PIL import Image

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
