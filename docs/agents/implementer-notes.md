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
- `agents/image_generator.py`: `optimise_prompt(request, topic, visual_style=None)` returns an `ImagePrompt` (`prompt` ending with `NO_TEXT_RULE`, `use`, `aspect`, `alt_text`); `aspect_of` picks the shape; `image_revision(...)` saves the PNG and returns the Revision with its `image` metadata; `image_message(piece_id, revision_number, metadata)` is the chat message, whose `additional_kwargs["content_piece"]` is `{"piece_id", "revision"}` (from 1). A refine step with a `target` goes to `_refine_image`: it re-optimises from the latest Revision's `image["prompt"]` plus the instructions, keeps `image["aspect"]` unless the optimiser returns one, and adds a *refinement* Revision through `add_revision` with its own file (`<piece>-r<n+1>.png`).
- `ImageMetadata` in `workflow/state.py` is the fixed record on `Revision["image"]`. Images have no Quality Score.
- `workflow/image_files.py`: `image_path(topic_id, piece_id, revision_number)` (`<data>/images/<topic>/<piece>-r<n>.png`) and `save_png`.
- Config: `image.models`, `image.sizes`, `image.quality`, `image.default_style`; `ImageOptions.size_for(model, aspect)` and `quality_for(model)`.
- Tests: `tests/conftest.py` blocks LiteLLM's image call and points the data folder at a temporary directory for every test.

**Model calls** (ADR 0002)
- Model calls go only through `integrations/llm.py`: `complete`, `complete_structured`, `complete_structured_reply`. After a second invalid reply they raise `StructuredOutputError`.
- Images: `generate_image(prompt, aspect)` uses the first image model; `generate_image_with(model, prompt, aspect)` one given model. Both return a `GeneratedImage(data, model, provider_url)` and raise `ImageGenerationError` for an unusable result.
- Each agent's model and fallbacks are in `config/models.yaml`. Thresholds are in its `quality` section.

## Conventions

- Match the existing style: short module docstrings that cite the spec, few comments, type hints, TypedDict/pydantic shapes, and ruff line length 100.
- Use glossary terms in names, tests and user-facing text.
- Keep echo mode working: with no `OPENAI_API_KEY`, the Query Handler echoes the message. The deployed app's smoke test relies on it.
