"""The Brand Profile store against a temporary data folder (spec #22, seam 2)."""

import json
import logging
from datetime import UTC, datetime

import pytest

from contentblitz.workflow.brand_profiles import (
    BrandProfileError,
    BrandProfileNotFound,
    BrandProfileStore,
)


def fields(**overrides):
    base = {
        "name": "Acme Health",
        "audience": "Hospital CIOs",
        "tone": ["confident", "warm"],
        "dos": ["use British spelling"],
        "donts": ["never make medical claims"],
        "words_to_avoid": ["revolutionary"],
        "visual_style": {
            "colours": ["#0A7CFF", "#FFFFFF"],
            "mood": "calm",
            "medium": "photo",
            "notes": "hospital corridors",
        },
    }
    return {**base, **overrides}


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "brand_profiles"


@pytest.fixture
def store(folder):
    return BrandProfileStore(folder)


def test_a_created_profile_survives_a_restart(folder, store):
    created = store.create(fields())

    reopened = BrandProfileStore(folder)

    assert created.id == "acme-health"
    assert [p.name for p in reopened.list()] == ["Default", "Acme Health"]
    got = reopened.get("acme-health")
    assert got.audience == "Hospital CIOs"
    assert got.visual_style.colours == ["#0A7CFF", "#FFFFFF"]
    assert got.created_at == created.created_at
    assert len(list(folder.glob("*.json"))) == 1


def test_update_refreshes_updated_at_and_keeps_the_id_and_created_at(folder):
    times = iter([datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 2, tzinfo=UTC)])
    store = BrandProfileStore(folder, clock=lambda: next(times))
    created = store.create(fields())

    updated = store.update("acme-health", fields(name="Acme Care", audience="Clinic owners"))

    assert updated.id == "acme-health"
    assert updated.created_at == datetime(2026, 10, 1, tzinfo=UTC)
    assert updated.updated_at == datetime(2026, 10, 2, tzinfo=UTC)
    reopened = BrandProfileStore(folder).get("acme-health")
    assert (reopened.name, reopened.audience) == ("Acme Care", "Clinic owners")
    assert reopened.created_at == created.created_at


def test_a_deleted_profile_is_gone(folder, store):
    store.create(fields())

    store.delete("acme-health")

    assert [p.id for p in BrandProfileStore(folder).list()] == ["default"]
    with pytest.raises(BrandProfileNotFound):
        store.get("acme-health")


def test_duplicate_makes_a_new_profile_under_a_unique_name(store):
    original = store.create(fields())

    first = store.duplicate("acme-health")
    second = store.duplicate("acme-health")

    assert (first.name, second.name) == ("Acme Health copy", "Acme Health copy 2")
    assert len({original.id, first.id, second.id}) == 3
    assert first.fields() | {"name": "Acme Health"} == original.fields()
    assert [p.name for p in store.list()][1:] == [
        "Acme Health",
        "Acme Health copy",
        "Acme Health copy 2",
    ]


def test_duplicate_takes_a_chosen_name(store):
    store.create(fields())

    copy = store.duplicate("acme-health", name="Acme Pharma")

    assert store.get(copy.id).name == "Acme Pharma"


def test_default_is_always_listed_first_and_never_written_to_disk(folder, store):
    store.create(fields(name="Aardvark Labs"))

    default = store.list()[0]

    assert (default.id, default.name) == ("default", "Default")
    assert default.tone and default.words_to_avoid == []
    assert store.get("default") == default
    assert [path.name for path in folder.iterdir()] == ["aardvark-labs.json"]


def test_default_lists_even_without_a_folder(folder):
    assert [p.id for p in BrandProfileStore(folder).list()] == ["default"]


def test_default_can_be_duplicated_but_not_updated_or_deleted(store):
    copy = store.duplicate("default")

    with pytest.raises(BrandProfileError):
        store.update("default", fields(name="Default"))
    with pytest.raises(BrandProfileError):
        store.delete("default")
    assert copy.name == "Default copy"
    assert copy.fields() | {"name": "Default"} == store.get("default").fields()


@pytest.mark.parametrize(
    ("bad", "field"),
    [
        ({"name": ""}, "name"),
        ({"name": "   "}, "name"),
        ({"name": "x" * 61}, "name"),
        ({"audience": "x" * 301}, "audience"),
        ({"tone": []}, "tone"),
        ({"tone": [f"tone {n}" for n in range(7)]}, "tone"),
        ({"tone": ["x" * 41]}, "tone"),
        ({"dos": ["x" * 201]}, "dos"),
        ({"writing_sample": "x" * 10_001}, "writing_sample"),
        ({"visual_style": {"colours": ["#12345"]}}, "visual_style"),
        ({"visual_style": {"colours": ["0A7CFF"]}}, "visual_style"),
        ({"visual_style": {"medium": "oil painting"}}, "visual_style"),
    ],
)
def test_an_invalid_profile_is_rejected_naming_the_field(store, bad, field):
    with pytest.raises(BrandProfileError) as error:
        store.create(fields(**bad))

    assert error.value.field.split(".")[0] == field
    assert field in str(error.value)
    assert [p.id for p in store.list()] == ["default"]


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"visual_style": {"colours": ["#123"]}}, "visual_style.colours.0: must be a #RRGGBB"),
        ({"name": " "}, "name: is required"),
        ({"tone": []}, "tone: needs at least 1"),
        ({"tone": ["t"] * 7}, "tone: can have at most 6"),
        ({"dos": ["x" * 201]}, "dos.0: must be at most 200 characters"),
    ],
)
def test_validation_errors_read_plainly(store, bad, message):
    with pytest.raises(BrandProfileError, match=f"^{message}"):
        store.create(fields(**bad))


def test_a_missing_name_is_rejected(store):
    without_name = {k: v for k, v in fields().items() if k != "name"}

    with pytest.raises(BrandProfileError) as error:
        store.create(without_name)

    assert error.value.field == "name"


@pytest.mark.parametrize("name", ["acme health", "ACME HEALTH", "default", "DEFAULT"])
def test_a_duplicate_name_in_any_case_is_rejected(store, name):
    store.create(fields())

    with pytest.raises(BrandProfileError) as error:
        store.create(fields(name=name))

    assert error.value.field == "name"


def test_update_rejects_another_profiles_name_but_keeps_its_own(store):
    store.create(fields())
    store.create(fields(name="Beta Corp"))

    with pytest.raises(BrandProfileError) as error:
        store.update("beta-corp", fields(name="Acme Health"))
    store.update("beta-corp", fields(name="BETA CORP"))

    assert error.value.field == "name"
    assert store.get("beta-corp").name == "BETA CORP"


def test_a_corrupt_or_invalid_file_is_skipped_with_a_warning(folder, store, caplog):
    store.create(fields())
    (folder / "broken.json").write_text("{not json", encoding="utf-8")
    (folder / "invalid.json").write_text('{"id": "invalid", "name": ""}', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        listed = store.list()

    assert [p.id for p in listed] == ["default", "acme-health"]
    assert "broken.json" in caplog.text
    assert "invalid.json" in caplog.text


def test_a_file_copied_by_hand_under_another_name_is_skipped(folder, store, caplog):
    store.create(fields())
    copied = (folder / "acme-health.json").read_text(encoding="utf-8")
    (folder / "acme-copy.json").write_text(copied, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        listed = store.list()

    assert [p.id for p in listed] == ["default", "acme-health"]
    assert "acme-copy.json" in caplog.text


def test_names_that_make_the_same_slug_get_distinct_ids(store):
    first = store.create(fields(name="Acme Health"))
    second = store.create(fields(name="Acme-Health!"))
    third = store.create(fields(name="Default!"))

    assert (first.id, second.id, third.id) == ("acme-health", "acme-health-2", "default-2")
    assert store.get("acme-health-2").name == "Acme-Health!"


def test_the_optional_writing_sample_and_voice_description_are_kept(folder, store):
    store.create(
        fields(writing_sample="We fix hospital IT.", voice_description="Short, direct sentences.")
    )
    plain = store.create(fields(name="Plain Co"))

    got = BrandProfileStore(folder).get("acme-health")

    assert got.writing_sample == "We fix hospital IT."
    assert got.voice_description == "Short, direct sentences."
    assert (plain.writing_sample, plain.voice_description) == (None, None)


def test_a_profile_file_is_readable_json(folder, store):
    store.create(fields())

    saved = json.loads((folder / "acme-health.json").read_text(encoding="utf-8"))

    assert saved["name"] == "Acme Health"
    assert saved["visual_style"]["medium"] == "photo"
    assert saved["created_at"].startswith("20")


def test_the_store_lives_under_the_configured_data_folder(tmp_path):
    BrandProfileStore().create(fields())

    assert (tmp_path / "data" / "brand_profiles" / "acme-health.json").is_file()


def test_update_accepts_an_edited_profile(store):
    profile = store.create(fields())

    store.update(profile.id, profile.model_copy(update={"audience": "Clinic owners"}))

    assert store.get(profile.id).audience == "Clinic owners"
