# ContentBlitz

A multi-agent content marketing assistant. Give it a Topic and it researches the web, builds a source-cited Research Brief, and turns it into SEO blogs, LinkedIn posts and images in your Brand Profile's voice.

> **Status:** skeleton. The LangGraph workflow, LiteLLM model layer, search fallback interface, Streamlit UI shell, tests, Docker and CI are in place. The Query Handler currently answers every message directly; the other five agents are stubs.

Vocabulary is defined in [GLOSSARY.md](GLOSSARY.md); key decisions are in [docs/adr/](docs/adr/).

## Architecture

```
            ┌──────────────────┐
 user ──▶   │  Query Handler   │  classifies intent, builds a plan
            └────────┬─────────┘
       ┌─────────────┼───────────────┬──────────────────┐
       ▼             ▼               ▼                  ▼
 Deep Research ─▶ Content       SEO Blog Writer   Image Generation
                  Strategist ─▶ LinkedIn Writer
                (Research Brief)  (draft → validate → revise ≤3)
```

| Concern | Choice |
|---|---|
| Orchestration | LangGraph |
| Model calls (text + image) | LiteLLM, models per agent in [`config/models.yaml`](config/models.yaml) |
| Research | SerpAPI, falling back to Tavily |
| UI | Streamlit |
| Tracing | LangSmith |
| Hosting | Docker → Azure Container Apps |

## Project layout

```
src/contentblitz/
  agents/        one module per agent
  core/          configuration
  integrations/  llm.py (LiteLLM), search.py
  workflow/      LangGraph state and graph
  web_app/       Streamlit UI
config/models.yaml   which model each agent uses, plus fallbacks
tests/           unit / integration / e2e (all providers mocked)
```

## Run locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt -e .
cp .env.example .env            # add OPENAI_API_KEY at minimum
.venv/bin/streamlit run src/contentblitz/web_app/streamlit_app.py
```

Without an API key the app still runs and echoes your messages, which is handy for checking a deployment.

## Run with Docker

```bash
docker compose up --build       # http://localhost:8501
```

Generated images and Brand Profiles are written to `./data`, mounted at `/data` in the container.

## Tests

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest                # prints coverage
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint, tests and a Docker build on every push. When CI passes on `main`, [deploy.yml](.github/workflows/deploy.yml) deploys to Azure Container Apps; see [Continuous deployment](docs/deployment_guide.md#continuous-deployment-from-github) for the one-time setup.

## Switching models

Edit [`config/models.yaml`](config/models.yaml). Each agent lists models in LiteLLM `provider/model` form; the first is used and the rest are fallbacks:

```yaml
blog_writer:
  models: [openai/gpt-4o, anthropic/claude-sonnet-4-20250514, gemini/gemini-2.0-flash]
```

Set the matching API key in `.env`.

## Deploy to Azure

See [docs/deployment_guide.md](docs/deployment_guide.md).
