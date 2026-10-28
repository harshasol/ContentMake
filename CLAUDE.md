# ContentBlitz

Multi-agent content marketing assistant (LangGraph + LiteLLM + Streamlit). Setup and layout: README.md. Vocabulary: GLOSSARY.md. Decisions: docs/adr/.

## Agent skills

### Issue tracker

GitHub Issues on harshasol/ContentMake via the `gh` CLI; one spec issue per feature, tickets as its sub-issues, closed by the feature PR. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
