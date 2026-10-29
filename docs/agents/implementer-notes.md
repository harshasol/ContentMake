# Implementer notes

Read this before building a ticket. It covers what the code doesn't make obvious: how to run the checks, how a feature moves from spec to `main`, and the interfaces that later features build on. The spec issues on GitHub hold the requirements, and `GLOSSARY.md` and `docs/adr/` hold the vocabulary and decisions.

## Running the checks

```bash
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pytest -q                 # full suite + coverage
.venv/bin/python -m pytest -q tests/integration/test_graph.py --no-cov   # one file, fast
```

- **Always use `python -m`.** If the virtualenv was created before the project folder was renamed, its `pytest`/`ruff` scripts point at the old path and won't run. Recreate it with `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt`.
- **The package is not pip-installed.** `pyproject.toml` sets `pythonpath = ["src"]`, so pytest tests the code of whichever checkout or worktree you run it from. In a git worktree, run the main checkout's `.venv/bin/python -m pytest` from the worktree root.
- **No network in tests.** Fake models at `integrations/llm.py` and search at `integrations/search.py`, either with monkeypatch or with httpx's `MockTransport`.

## From spec to `main`

1. **Spec:** `/to-spec` writes one spec issue per feature.
2. **Tickets:** `/to-tickets` creates them as sub-issues of the spec, with native blocking links. Every user story in the spec maps to one ticket. Each ticket lists its stories under "User stories covered", and the spec has a "Story coverage" comment.
3. **Building:** `/implement-spec #<spec>` creates `feature/<slug>`. Each ticket is built test-first on `ticket/<n>-<slug>` in its own worktree, then merged into the feature branch.
4. **Review:** `/code-review` runs on the feature branch, and the fixes go in one pass.
5. **Merge:** one PR per feature closes the spec and its tickets. `main` is protected (PR required, and CI jobs `test` and `docker` must pass), and merging deploys to Azure (`docs/deployment_guide.md`).

Commit messages end with the `Co-Authored-By` trailer given in the session.

## Interfaces later features build on

**Routing and plans** (#4)
- `workflow/routing.py`, `RoutingDecision`: `intent` (`new_content` / `refine` / `chat` / `unclear`), `topic_name`, `same_topic`, `formats`, `fresh_research`, `refine_target`, `instructions`, the writer options `target_keyword` / `template` / `length` / `length_unit`, `source_piece` / `source_kind` (what a new piece builds on), `reply`, `clarifying_question`, `topic_answer`. The glossary avoids "subject", so use Topic terms.
- **A narrower angle on the active Topic asks the Topic question.** Only a request on the same Topic runs straight away.
- `workflow/formats.py` is the single format table. Each format maps to `FormatSpec(agent, label, kind, options, templates, length_unit)`: `options` are the routing fields its writer step takes, and `plan.writer_options` reads them. Add a new format here, not in parallel maps. Content Pieces keep the field name `kind`.
- `workflow/plan.py` builds plans in code: `build_plan`, `describe_plan`, `replan`. The research-first rule (ADR 0001) lives only in `build_plan`.
- `workflow/graph.py` runs the dispatcher loop. An agent stops the plan by returning `{"abort_reason": reason}`. A writer that fails posts an error, and the plan carries on.

**State and Topics**
- `workflow/state.py` holds `Topic`, `ResearchRecord`/`ResearchSource`, `ResearchBrief`, `ContentPiece`, `Revision`, `PlanStep` and `PendingTopicQuestion`.
- `workflow/topics.py` holds `active_topic`, `create_topic`, `new_piece_id`, `add_piece`, `add_revision`, `piece_topic` and `piece_title`. **Always save Content Pieces and Revisions through `add_piece` and `add_revision`.** Refine finds pieces through each Topic's `piece_ids`, so writing state directly breaks refinement.
- `workflow/progress.py` holds `progress_message` and `is_progress`. Agents post progress through it, which keeps those messages out of the router's ~10-message history window.

**Research and Brief** (#5, #6)
- `integrations/search.py` holds `configured_providers`, `missing_key_names`, `search_with_fallback` and `fetch_page`.
- `workflow/sources.py` holds `merge_sources` and `normalise_url`.
- `workflow/statistics.py` has `is_verified`, and `figures` / `is_grounded`, which LinkedIn's "grounded figures" check uses with the same number matching.
- `workflow/report.py` holds `render_report` and `summary_line`. The Research tab and export (#23) reuse them.

**Quality loop** (#19, reused by #20 and later formats)
- `workflow/quality_loop.py`: a format plugs in as a `QualityFormat(name, draft, revise, checks, judge, finalise, failure)`, and `quality_loop(fmt)` builds the writer's node. `agents/blog_writer.py` (`BLOG = QualityFormat(...)`) and `agents/linkedin_writer.py` (`LINKEDIN`) are the worked examples, each with a `*_REFINEMENT` variant. `quality_text` is the Quality Score block every format's chat message ends with.
- `workflow/content_checks.py` provides the `Draft`, `Post` and `CheckResult` shapes and one function per check. `check_blog` and `check_linkedin` compose them. A new format adds its own `check_<format>`.
- `workflow/quality_score.py` holds `quality_score` and `comparable_points`: checks are worth 70 points and the reviewer 30.
- `agents/quality_judge.py` is the AI reviewer, using the `quality_judge` model in `config/models.yaml`.

**Images** (#21)
- An image step can carry `source_piece` (the image format takes it as an option, resolved by `resolve_source` as for repurposing): `optimise_prompt(..., piece=...)` gets that piece's title and kind, its kind sets the use (`FormatSpec.image_use` in the format table), and the metadata records `ImageMetadata["source_piece"]` (`NotRequired`, absent without a source).
- `agents/image_generator.py`: `optimise_prompt(request, topic, visual_style=None, piece=None)` returns an `ImagePrompt` (`prompt` ending with `NO_TEXT_RULE`, `use`, `aspect`, `alt_text`); `aspect_of` picks the shape; `image_revision(...)` saves the PNG and returns the Revision with its `image` metadata; `image_message(piece_id, revision_number, metadata)` is the chat message, whose `additional_kwargs["content_piece"]` is `{"piece_id", "revision"}` (from 1).
- New and refined images share one path: `_new_image` / `_refine_image` each build an `_ImageJob` (request, optimiser request, source piece, fallback prompt, kept aspect, save function) and `_make_image` runs it. Change the shared steps there, not per path.
- A refine step with a `target` goes to `_refine_image`: it re-optimises from the latest Revision's `image["prompt"]` plus the instructions, gives the optimiser the piece in `image["source_piece"]` again (looked up on the Topic), keeps `image["aspect"]` and any `image["source_piece"]` unless the optimiser returns a new shape, and adds a *refinement* Revision through `add_revision` with its own file (`<piece>-r<n+1>.png`).
- A failed prompt optimisation doesn't stop the image: it is drawn from a plain prompt (the request, or the latest prompt plus the instructions, with `NO_TEXT_RULE`), in the shape `asked_aspect(request)` finds, otherwise the source piece's or latest Revision's, otherwise square. Only a failure after that (saving the file) posts `NO_IMAGE` / `NO_REFINED_IMAGE`.
- `make_image(prompt, aspect, topic_name, colour=None)` returns a `GeneratedImage` and never raises for a failed model: it falls back to a placeholder card (model `PLACEHOLDER_MODEL`, in `colour` or `image.placeholder_colour`).
- **Brand visual style (#55).** `_make_image` reads `active_brand_profile(state).visual_style` for new and refined images alike: `describe_visual_style(style)` (colours with the primary named, mood, medium, notes) is the optimiser's `visual_style`, and `colours[0]` (or `None`) is `make_image`'s `colour`. `image.default_style` in config is only used when `optimise_prompt` is called without a style. Image tests pass `brand_profile_id=` to `run_plan`/`run_refine` in `tests/integration/test_image_generator.py`, which load it through `load_brand_profile`. **Generate images through it, not `llm.generate_image`**, so new and refined images both get the placeholder. `image_revision` sets the `placeholder` flag and `fallbacks` from the `GeneratedImage`, and `image_message` (an `AIMessage`) says when it is a placeholder. The `image` trace has `request`, `optimised_prompt`, `optimiser_failure`, `models_tried` (`GeneratedImage.models_tried()`: the failed models, then the one used, `placeholder` included) and `placeholder`.
- `ImageMetadata` in `workflow/state.py` is the fixed record on `Revision["image"]`. Images have no Quality Score.
- The Chat tab (`web_app/streamlit_app.py`) renders any assistant message with a `content_piece` reference (looked up with `find_revision(state, piece_id, number)` in `workflow/topics.py`) as the image loaded from `file_path`, a caption (`image_caption`: `model · size`, or `Placeholder image · size` when `placeholder` is set), and an "Optimised prompt" expander. A refined image just needs its message to reference the new Revision.
- `workflow/image_files.py`: `image_path(topic_id, piece_id, revision_number)` (`<data>/images/<topic>/<piece>-r<n>.png`) and `save_png`.
- `workflow/image_files.py` also draws the card: `placeholder_png(size, colour, topic_name)`, and owns the chat's wording: `PLACEHOLDER_LABEL`, `size_label(metadata)` (`1536×1024`) and `image_caption(metadata)`. Use them rather than formatting sizes or the label again.
- Config: `image.models`, `image.sizes`, `image.quality`, `image.default_style`, `image.placeholder_colour`; `ImageOptions.size_for(model, aspect)` and `quality_for(model)`. `HEX_COLOUR` in `core/config.py` is the one `#RRGGBB` pattern.
- Tests: `tests/conftest.py` blocks LiteLLM's image call and points the data folder at a temporary directory for every test.

**Brand Profiles** (#22)
- `workflow/brand_profiles.py`: `BrandProfileStore(folder=None, clock=...)` (by default `<data>/brand_profiles/`, one `<id>.json` per profile) with `list()` (Default first, then by name; bad files skipped with a warning), `get(id)`, `create(fields)`, `update(id, fields)`, `delete(id)` and `duplicate(id, name=None)` (named "<name> copy", numbered until unique). `fields` is a dict or a `BrandProfileFields`/`BrandProfile`; `profile.fields()` gives the editable fields back.
- `BrandProfile` (pydantic) has `id` (a slug that never changes on rename), `name`, `audience`, `tone` (1–6), `dos`, `donts`, `words_to_avoid`, `writing_sample`, `voice_description` (both optional), `visual_style: VisualStyle(colours, mood, medium, notes)` (`colours[0]` is the primary; `medium` is `photo` / `illustration` / `3D` / `flat graphic`), `created_at`, `updated_at` (UTC).
- `DEFAULT_PROFILE` / `DEFAULT_PROFILE_ID` (`"default"`): built in, never a file. `update`/`delete` on it raise; `duplicate` works.
- Errors: `BrandProfileError` with `.field` (a dotted path such as `name` or `visual_style.colours.0`; `None` for the Default guard) and a plain message, e.g. `"tone: can have at most 6"`. `get` raises `BrandProfileNotFound` (a `BrandProfileError`) for an unknown or unreadable id; fall back to Default there.
- **The active Brand Profile (#52).** The sidebar passes `brand_profile_id` into every run as conversation input (`st.session_state.brand_profile_id`, `"default"` for a new conversation). The graph's first node, `load_brand_profile` (`workflow/active_brand_profile.py`), loads it through the store into `state["brand_profile"]` (a JSON dump). A missing id means Default; an id with no saved profile falls back to Default and posts a one-line progress notice once per conversation (`brand_profile_fallback_noticed`).
- **Inside an agent, read the profile with `active_brand_profile(state) -> BrandProfile`** (from `workflow/active_brand_profile.py`). Never read the store or the UI from an agent, and never read `state["brand_profile"]` directly. It returns `DEFAULT_PROFILE` when nothing was loaded (an agent unit-tested without the graph), so writers, the reviewer and images (#53–#55) always get a profile.
- **Writers (#53).** `brand_voice(state) -> str` (`workflow/brand_voice.py`) is the active profile's audience, tone, do's, don'ts, words to avoid and learned voice description (never the writing sample) as prompt text. Both writers' `_prompt` include it, so drafts, automatic revisions and refinements all get it. Research, the Content Strategist and the Query Handler don't. Tests: `record_model_calls(monkeypatch)` (after `fake_router`) records every structured call as `(agent, prompt text)`; `prompts_to(calls, agent)` filters them.
- **Revisions record it automatically.** `add_piece` and `add_revision` stamp each new Revision with `brand_profile: BrandProfileRecord` (`{id, name, updated_at}`, from `brand_profile_record(state)`), unless it already has one. Don't set it in the agents. Existing Revisions are never touched, so editing or switching a profile only affects new pieces and refinements.
- **The Brand Profiles tab (#56)** is `web_app/brand_profiles_tab.py` (`brand_profiles_tab()`). The form is plain widgets (not `st.form`) keyed `bp_<field>` (`bp_name`, `bp_tone`, ..., `bp_colours`, `bp_medium`); list fields are one per line, colours comma-separated. `fill_form(fields)` puts `BrandProfile.fields()`-shaped values (any subset) into them, and `form_fields()` reads them back for the store. Every action is a button `on_click` callback, so the sidebar picker sees a change on the same rerun; **fill fields from a callback** (e.g. a "Learn voice" button, #57), never after the widgets are drawn. `voice_description` is carried in `st.session_state.bp_voice_description` with no widget yet. Store errors land in `st.session_state.bp_error = (field, message)` (`BrandProfileError.message` is the text without the field) and show under that field. `BrandProfileStore.copy_name(name)` gives the free "<name> copy" name.
- Tests: set the profile per run with `graph.invoke({"messages": [...], "brand_profile_id": id}, ...)` (`send_as` in `tests/integration/test_graph.py`); create profiles with `BrandProfileStore()`, which uses the temporary data folder from `tests/conftest.py`. Revisions made in a Default conversation record `{"id": "default", "name": "Default", "updated_at": "2026-10-29T00:00:00+00:00"}`, so exact-match Revision assertions include it.

**Model calls** (ADR 0002)
- Model calls go only through `integrations/llm.py`: `complete`, `complete_structured`, `complete_structured_reply`. After a second invalid reply they raise `StructuredOutputError`.
- Images: `generate_image(prompt, aspect)` tries the image models in config order, skipping one whose provider has no API key, and rejects a result that fails `check_image` (PNG/JPEG/WebP, both sides ≥ 512 px, not nearly one colour across the RGB channels). It returns a `GeneratedImage(data, model, provider_url, fallbacks)`, where `fallbacks` is a tuple of `ModelOutcome` (`{"model", "outcome"}`, also what `ImageMetadata["fallbacks"]` holds), or raises `NoUsableImage` (with `.fallbacks`) when every model fails. `generate_image_with(model, prompt, aspect)` calls one model, unchecked, and raises `ImageGenerationError` for no image.
- Each agent's model and fallbacks are in `config/models.yaml`. Thresholds are in its `quality` section.

## Conventions

- Match the existing style: short module docstrings that cite the spec, few comments, type hints, TypedDict/pydantic shapes, and ruff line length 100.
- Use glossary terms in names, tests and user-facing text.
- Keep echo mode working: with no `OPENAI_API_KEY`, the Query Handler echoes the message. The deployed app's smoke test relies on it.
