"""The conversation's active Brand Profile: loaded at the start of each run, read by agents.

The sidebar passes the active Brand Profile's id in as conversation input (`brand_profile_id`);
agents never read the UI. `load_brand_profile`, the graph's first node, loads it through the
store, and agents read it with `active_brand_profile(state)`. See spec #22.
"""

from contentblitz.workflow.brand_profiles import (
    DEFAULT_PROFILE,
    BrandProfile,
    BrandProfileNotFound,
    BrandProfileStore,
)
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.state import BrandProfileRecord, ContentBlitzState

MISSING_PROFILE = "Brand Profile *{name}* no longer exists, so this conversation uses Default."


def active_brand_profile(state: ContentBlitzState) -> BrandProfile:
    """The Brand Profile this run uses; Default when none has been loaded."""
    loaded = state.get("brand_profile")
    return BrandProfile.model_validate(loaded) if loaded else DEFAULT_PROFILE


def brand_profile_record(state: ContentBlitzState) -> BrandProfileRecord:
    """What a new Revision records about the active Brand Profile: its id, name and version."""
    profile = active_brand_profile(state)
    return {"id": profile.id, "name": profile.name, "updated_at": profile.updated_at.isoformat()}


def load_brand_profile(state: ContentBlitzState) -> ContentBlitzState:
    """The graph's first node: loads the active Brand Profile into the state for this run.

    No id means Default. An id with no saved profile (it was deleted) also means Default, and
    the first time that happens for that id in the conversation the chat gets a one-line notice.
    """
    profile_id = state.get("brand_profile_id")
    try:
        profile = BrandProfileStore().get(profile_id) if profile_id else DEFAULT_PROFILE
    except BrandProfileNotFound:
        return {"brand_profile": DEFAULT_PROFILE.model_dump(mode="json"), **_notice(state)}
    return {"brand_profile": profile.model_dump(mode="json")}


def _notice(state: ContentBlitzState) -> ContentBlitzState:
    profile_id = state["brand_profile_id"]
    noticed = state.get("missing_brand_profiles_noticed") or []
    if profile_id in noticed:
        return {}
    previous = state.get("brand_profile") or {}
    name = previous["name"] if previous.get("id") == profile_id else profile_id
    return {
        "messages": [progress_message(MISSING_PROFILE.format(name=name))],
        "missing_brand_profiles_noticed": [*noticed, profile_id],
    }
