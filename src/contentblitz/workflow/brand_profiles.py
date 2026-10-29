"""The Brand Profile store: saved Brand Profiles plus a built-in, read-only Default.

Each saved profile is one JSON file in a Brand Profiles folder under the data folder (`/data` on
Azure, ADR 0003; a single replica means no concurrent writers). The sidebar, the workflow and
the Brand Profiles tab all read and write profiles through `BrandProfileStore`. See spec #22.
"""

import logging
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from contentblitz.core.config import HEX_COLOUR, get_settings

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_ID = "default"
FOLDER_NAME = "brand_profiles"
NAME_MAX_LENGTH = 60

Medium = Literal["photo", "illustration", "3D", "flat graphic"]


def _text(max_length: int, min_length: int = 0) -> Any:
    return Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=min_length, max_length=max_length),
    ]


Rule = _text(200, min_length=1)
Colour = Annotated[str, StringConstraints(pattern=f"^{HEX_COLOUR}$")]


class VisualStyle(BaseModel):
    """How the brand's images look: colours (the first is the primary), mood, medium, notes."""

    model_config = ConfigDict(extra="forbid")

    colours: list[Colour] = Field(default_factory=list, max_length=6)
    mood: _text(200) = ""
    medium: Medium = "illustration"
    notes: _text(500) = ""


class BrandProfileFields(BaseModel):
    """What the marketer fills in for a Brand Profile; the store adds the id and timestamps."""

    model_config = ConfigDict(extra="forbid")

    name: _text(NAME_MAX_LENGTH, min_length=1)
    audience: _text(300) = ""
    tone: list[_text(40, min_length=1)] = Field(min_length=1, max_length=6)
    dos: list[Rule] = Field(default_factory=list, max_length=20)
    donts: list[Rule] = Field(default_factory=list, max_length=20)
    words_to_avoid: list[_text(60, min_length=1)] = Field(default_factory=list, max_length=50)
    writing_sample: _text(10_000) | None = None
    voice_description: _text(1_000) | None = None
    visual_style: VisualStyle = Field(default_factory=VisualStyle)


class BrandProfile(BrandProfileFields):
    """A saved (or the built-in Default) Brand Profile. Timestamps are UTC."""

    id: str
    created_at: datetime
    updated_at: datetime

    def fields(self) -> dict[str, Any]:
        """The editable fields, ready to pass back to `create` or `update`."""
        return self.model_dump(include=set(BrandProfileFields.model_fields))


_BUILT_IN = datetime(2026, 10, 29, tzinfo=UTC)

DEFAULT_PROFILE = BrandProfile(
    id=DEFAULT_PROFILE_ID,
    name="Default",
    audience="A general business audience",
    tone=["clear", "professional", "helpful"],
    dos=["explain jargon in plain words", "back claims with evidence"],
    donts=["overstate claims"],
    visual_style=VisualStyle(
        colours=["#4A5568"],
        mood="calm and professional",
        medium="illustration",
        notes="clean and modern, with plenty of negative space",
    ),
    created_at=_BUILT_IN,
    updated_at=_BUILT_IN,
)


class BrandProfileError(ValueError):
    """A Brand Profile can't be saved or changed; `field` names the field at fault, if any."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(f"{field}: {message}" if field else message)
        self.field = field


class BrandProfileNotFound(BrandProfileError):
    """No Brand Profile has this id."""


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "profile"


def _validated(fields: Mapping[str, Any] | BrandProfileFields) -> BrandProfileFields:
    if isinstance(fields, BaseModel):
        fields = fields.model_dump(include=set(BrandProfileFields.model_fields))
    try:
        return BrandProfileFields.model_validate(dict(fields))
    except ValidationError as error:
        first = error.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or None
        raise BrandProfileError(_plain_message(first), field) from error


def _plain_message(error: Any) -> str:
    ctx = error.get("ctx", {})
    match error["type"]:
        case "missing":
            return "is required"
        case "string_pattern_mismatch":
            return "must be a #RRGGBB hex colour, like #0A7CFF"
        case "string_too_short":
            return "is required"
        case "string_too_long":
            return f"must be at most {ctx['max_length']} characters"
        case "too_short":
            return f"needs at least {ctx['min_length']}"
        case "too_long":
            return f"can have at most {ctx['max_length']}"
        case _:
            return error["msg"]


class BrandProfileStore:
    """List, get, create, update, delete and duplicate Brand Profiles in `folder` (by default
    the Brand Profiles folder under the configured data folder)."""

    def __init__(
        self, folder: Path | None = None, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ):
        self.folder = folder or get_settings().contentblitz_data_dir / FOLDER_NAME
        self._clock = clock

    def list(self) -> list[BrandProfile]:
        """Default first, then the saved profiles by name. Bad files are skipped and logged."""
        saved = [p for p in map(self._load, sorted(self.folder.glob("*.json"))) if p]
        return [DEFAULT_PROFILE, *sorted(saved, key=lambda p: p.name.casefold())]

    def get(self, profile_id: str) -> BrandProfile:
        if profile_id == DEFAULT_PROFILE_ID:
            return DEFAULT_PROFILE
        path = self._path(profile_id)
        profile = self._load(path) if path.is_file() else None
        if profile is None:
            raise BrandProfileNotFound(f"No Brand Profile with id {profile_id!r}")
        return profile

    def create(self, fields: Mapping[str, Any] | BrandProfileFields) -> BrandProfile:
        valid = _validated(fields)
        self._check_name_free(valid.name)
        now = self._clock()
        profile = BrandProfile(
            **valid.model_dump(), id=self._new_id(valid.name), created_at=now, updated_at=now
        )
        self._save(profile)
        return profile

    def update(
        self, profile_id: str, fields: Mapping[str, Any] | BrandProfileFields
    ) -> BrandProfile:
        """Replace a saved profile's fields. Its id and created at stay; updated at is now."""
        current = self._editable(profile_id)
        valid = _validated(fields)
        self._check_name_free(valid.name, own_id=profile_id)
        profile = BrandProfile(
            **valid.model_dump(),
            id=current.id,
            created_at=current.created_at,
            updated_at=self._clock(),
        )
        self._save(profile)
        return profile

    def delete(self, profile_id: str) -> None:
        self._editable(profile_id)
        self._path(profile_id).unlink()

    def duplicate(self, profile_id: str, name: str | None = None) -> BrandProfile:
        """A new profile with `profile_id`'s fields, under `name` or "<name> copy" (numbered
        until unique). Default can be duplicated."""
        source = self.get(profile_id)
        return self.create({**source.fields(), "name": name or self._copy_name(source.name)})

    def _copy_name(self, name: str) -> str:
        taken = {profile.name.casefold() for profile in self.list()}
        n = 1
        while True:
            suffix = " copy" if n == 1 else f" copy {n}"
            candidate = name[: NAME_MAX_LENGTH - len(suffix)].rstrip() + suffix
            if candidate.casefold() not in taken:
                return candidate
            n += 1

    def _editable(self, profile_id: str) -> BrandProfile:
        if profile_id == DEFAULT_PROFILE_ID:
            raise BrandProfileError("the Default Brand Profile can't be changed; duplicate it")
        return self.get(profile_id)

    def _check_name_free(self, name: str, own_id: str | None = None) -> None:
        for profile in self.list():
            if profile.id != own_id and profile.name.casefold() == name.casefold():
                raise BrandProfileError(f"a Brand Profile called {profile.name!r} exists", "name")

    def _new_id(self, name: str) -> str:
        base = slugify(name)
        candidate, n = base, 2
        while candidate == DEFAULT_PROFILE_ID or self._path(candidate).exists():
            candidate, n = f"{base}-{n}", n + 1
        return candidate

    def _path(self, profile_id: str) -> Path:
        return self.folder / f"{slugify(profile_id)}.json"

    def _save(self, profile: BrandProfile) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._path(profile.id).write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def _load(self, path: Path) -> BrandProfile | None:
        try:
            profile = BrandProfile.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            logger.warning("Skipping Brand Profile file %s: %s", path.name, error)
            return None
        if profile.id != path.stem:
            logger.warning("Skipping Brand Profile file %s: its id is %r", path.name, profile.id)
            return None
        return profile
