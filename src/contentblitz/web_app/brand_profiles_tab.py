"""The Brand Profiles tab: list, create, edit, duplicate and delete Brand Profiles (#22, #56).

Every action runs in a button callback, before the script reruns, so the sidebar picker shows
the change on that same rerun. The form is plain widgets keyed `bp_<field>`; `fill_form` sets
them from a callback, which is how a button can fill fields before the marketer saves.
"""

import re
from collections.abc import Mapping
from typing import Any, get_args

import streamlit as st

from contentblitz.workflow.brand_profiles import (
    DEFAULT_PROFILE_ID,
    BrandProfile,
    BrandProfileError,
    BrandProfileFields,
    BrandProfileStore,
    Medium,
    VisualStyle,
)

MEDIUMS = list(get_args(Medium))
LIST_FIELDS = ("tone", "dos", "donts", "words_to_avoid")
STYLE_FIELDS = ("colours", "mood", "medium", "notes")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def fill_form(fields: Mapping[str, Any]) -> None:
    """Put Brand Profile fields (as `BrandProfile.fields()` gives them, or any subset) into the
    form's widgets. Call it from a callback, before the form is drawn."""
    state = st.session_state
    style = fields.get("visual_style") or {}
    values = {**{k: v for k, v in fields.items() if k != "visual_style"}, **style}
    for name, value in values.items():
        if name in LIST_FIELDS:
            value = "\n".join(value)
        elif name == "colours":
            value = ", ".join(value)
        elif value is None:
            value = ""
        state[f"bp_{name}"] = value


def form_fields() -> dict[str, Any]:
    """The form's widgets as Brand Profile fields, ready for the store to validate."""
    state = st.session_state

    def text(name: str) -> str:
        return state.get(f"bp_{name}", "")

    return {
        "name": text("name"),
        "audience": text("audience"),
        **{name: _lines(text(name)) for name in LIST_FIELDS},
        "writing_sample": text("writing_sample").strip() or None,
        "voice_description": text("voice_description").strip() or None,
        "visual_style": {
            "colours": [c for c in re.split(r"[,\s]+", text("colours")) if c],
            "mood": text("mood"),
            "medium": state.get("bp_medium", VisualStyle().medium),
            "notes": text("notes"),
        },
    }


def _open_form(profile_id: str | None, fields: Mapping[str, Any]) -> None:
    st.session_state.bp_editing = profile_id
    st.session_state.bp_form_open = True
    st.session_state.bp_error = None
    fill_form(fields)


def _new() -> None:
    blank = BrandProfileFields.model_construct(name="", tone=[], visual_style=VisualStyle())
    _open_form(None, blank.model_dump())


def _edit(profile_id: str) -> None:
    _open_form(profile_id, BrandProfileStore().get(profile_id).fields())


def _duplicate(profile_id: str) -> None:
    store = BrandProfileStore()
    source = store.get(profile_id)
    _open_form(None, {**source.fields(), "name": store.copy_name(source.name)})


def _ask_delete(profile_id: str) -> None:
    st.session_state.bp_deleting = profile_id


def _delete() -> None:
    profile_id = st.session_state.pop("bp_deleting", None)
    if profile_id:
        BrandProfileStore().delete(profile_id)
    if profile_id and st.session_state.get("bp_editing") == profile_id:
        _close()


def _cancel_delete() -> None:
    st.session_state.bp_deleting = None


def _save() -> None:
    store = BrandProfileStore()
    editing = st.session_state.get("bp_editing")
    try:
        if editing:
            store.update(editing, form_fields())
        else:
            store.create(form_fields())
    except BrandProfileError as error:
        st.session_state.bp_error = (error.field, error.message)
        return
    _close()


def _close() -> None:
    st.session_state.bp_form_open = False
    st.session_state.bp_editing = None
    st.session_state.bp_error = None


def _field_error(name: str) -> None:
    """The store's error, shown under the field it is about."""
    error = st.session_state.get("bp_error")
    if not error or not error[0]:
        return
    path = error[0].split(".")
    at = path[1] if path[0] == "visual_style" and len(path) > 1 else path[0]
    if at == name:
        st.error(error[1])


def _form() -> None:
    editing = st.session_state.get("bp_editing")
    st.subheader("Edit Brand Profile" if editing else "New Brand Profile")
    error = st.session_state.get("bp_error")
    if error and not error[0]:
        st.error(error[1])
    st.text_input("Name", key="bp_name")
    _field_error("name")
    st.text_input("Target audience", key="bp_audience")
    _field_error("audience")
    st.text_area("Tone descriptors (one per line, 1–6)", key="bp_tone")
    _field_error("tone")
    st.text_area("Do's (one per line)", key="bp_dos")
    _field_error("dos")
    st.text_area("Don'ts (one per line)", key="bp_donts")
    _field_error("donts")
    st.text_area("Words to avoid (one per line)", key="bp_words_to_avoid")
    _field_error("words_to_avoid")
    st.text_area("Writing sample (optional)", key="bp_writing_sample")
    _field_error("writing_sample")
    _field_error("voice_description")
    st.text_input("Colours (hex codes, primary first)", key="bp_colours")
    _field_error("colours")
    st.text_input("Mood", key="bp_mood")
    _field_error("mood")
    st.selectbox("Medium", MEDIUMS, key="bp_medium")
    _field_error("medium")
    st.text_area("Notes", key="bp_notes")
    _field_error("notes")
    save, cancel = st.columns(2)
    save.button("Save", key="bp_save", type="primary", on_click=_save)
    cancel.button("Cancel", key="bp_cancel", on_click=_close)


def _profile_row(profile: BrandProfile) -> None:
    built_in = profile.id == DEFAULT_PROFILE_ID
    updated = (
        "built in, read-only" if built_in else f"updated {profile.updated_at:%Y-%m-%d %H:%M} UTC"
    )
    st.markdown(f"**{profile.name}** · {profile.audience or 'no audience set'} · {updated}")
    if st.session_state.get("bp_deleting") == profile.id:
        st.warning(f"Delete **{profile.name}**? This can't be undone.")
        confirm, cancel = st.columns(2)
        confirm.button("Delete", key="bp_confirm_delete", type="primary", on_click=_delete)
        cancel.button("Keep it", key="bp_cancel_delete", on_click=_cancel_delete)
        return
    edit, duplicate, delete = st.columns(3)
    args = (profile.id,)
    if not built_in:
        edit.button("Edit", key=f"bp_edit_{profile.id}", on_click=_edit, args=args)
    duplicate.button("Duplicate", key=f"bp_duplicate_{profile.id}", on_click=_duplicate, args=args)
    if not built_in:
        delete.button("Delete", key=f"bp_delete_{profile.id}", on_click=_ask_delete, args=args)


def brand_profiles_tab() -> None:
    """The Brand Profiles tab's contents."""
    for profile in BrandProfileStore().list():
        _profile_row(profile)
    st.divider()
    st.button("New Brand Profile", key="bp_new", on_click=_new)
    if st.session_state.get("bp_form_open"):
        _form()
