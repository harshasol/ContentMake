"""Application configuration: secrets from the environment, model routing from YAML."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

AGENT_NAMES = (
    "query_handler",
    "research",
    "content_strategist",
    "blog_writer",
    "linkedin_writer",
    "image_generator",
    "quality_judge",
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "models.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    serpapi_api_key: str | None = None
    tavily_api_key: str | None = None
    langsmith_tracing: bool = False

    contentblitz_config: Path = DEFAULT_CONFIG_PATH
    contentblitz_data_dir: Path = Path("data")


class AgentModels(BaseModel):
    models: list[str] = Field(min_length=1)

    @property
    def primary(self) -> str:
        return self.models[0]

    @property
    def fallbacks(self) -> list[str]:
        return self.models[1:]


class LLMOptions(BaseModel):
    num_retries: int = 2
    timeout_seconds: int = 60


class BlogQuality(BaseModel):
    """The rules a blog is written and checked against."""

    word_range: tuple[int, int] = (900, 1800)
    density_range: tuple[float, float] = (1.0, 2.0)  # target keyword share of words, in %
    meta_length_range: tuple[int, int] = (150, 160)  # meta description length, in characters
    max_grade: float = 10  # highest Flesch-Kincaid grade level
    min_citations: int = 3  # fewest distinct Sources cited


CTA_PHRASES = (
    "comment",
    "share",
    "follow",
    "thoughts",
    "let me know",
    "dm",
    "link in",
    "repost",
    "tag someone",
    "join",
    "sign up",
    "register",
)


class LinkedInQuality(BaseModel):
    """The rules a LinkedIn post is written and checked against."""

    char_range: tuple[int, int] = (1300, 1600)  # the whole post, hashtags included
    hashtag_range: tuple[int, int] = (8, 12)
    max_hook_chars: int = 150  # the first line, which shows before "...see more"
    min_paragraphs: int = 4  # separated by blank lines, not counting the hashtag line
    max_paragraph_chars: int = 300
    # The last paragraph is a call to action if it ends with a question or has one of these.
    cta_phrases: list[str] = Field(default_factory=lambda: list(CTA_PHRASES))


class QualityOptions(BaseModel):
    max_revisions: int = 3
    pass_mark: float = 7  # the reviewer's average score (1-10) a draft needs to pass
    blog: BlogQuality = BlogQuality()
    linkedin: LinkedInQuality = LinkedInQuality()


class SearchOptions(BaseModel):
    """Search providers in fallback order, and the caps on each research run."""

    providers: list[Literal["serpapi", "tavily"]] = Field(min_length=1)
    queries_per_run: int = Field(4, ge=1)
    max_sources: int = Field(10, ge=1)
    pages_to_read: int = Field(6, ge=0)
    page_timeout_seconds: float = Field(8, gt=0)
    text_cap_chars: int = Field(2000, ge=0)


Aspect = Literal["landscape", "square", "portrait"]

# Used for an image model the size map doesn't list (gpt-image-1's sizes).
DEFAULT_IMAGE_SIZES: dict[Aspect, str] = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}
DEFAULT_IMAGE_QUALITY = "medium"
DEFAULT_VISUAL_STYLE = (
    "clean, modern editorial illustration with soft natural light, a calm professional "
    "colour palette and plenty of negative space"
)
DEFAULT_PLACEHOLDER_COLOUR = "#4A5568"  # a neutral slate
HEX_COLOUR = r"^#[0-9A-Fa-f]{6}$"


class ImageOptions(AgentModels):
    """The image models in fallback order, the size each makes for an aspect, its quality, the
    visual style used when no Brand Profile gives one, and the placeholder card's colour when
    the visual style gives none. See spec #21."""

    sizes: dict[str, dict[Aspect, str]] = {}  # model -> aspect -> "WIDTHxHEIGHT"
    quality: dict[str, str] = {}  # model -> the quality it is asked for
    default_style: str = DEFAULT_VISUAL_STYLE
    placeholder_colour: str = Field(DEFAULT_PLACEHOLDER_COLOUR, pattern=HEX_COLOUR)  # "#RRGGBB"

    def size_for(self, model: str, aspect: Aspect) -> str:
        return {**DEFAULT_IMAGE_SIZES, **self.sizes.get(model, {})}[aspect]

    def quality_for(self, model: str) -> str:
        return self.quality.get(model, DEFAULT_IMAGE_QUALITY)


class ModelConfig(BaseModel):
    agents: dict[str, AgentModels]
    image: ImageOptions
    search: SearchOptions
    llm: LLMOptions = LLMOptions()
    quality: QualityOptions = QualityOptions()

    def for_agent(self, agent: str) -> AgentModels:
        if agent not in self.agents:
            raise KeyError(f"No models configured for agent '{agent}'")
        return self.agents[agent]


def load_model_config(path: Path) -> ModelConfig:
    config = ModelConfig.model_validate(yaml.safe_load(path.read_text()))
    missing = set(AGENT_NAMES) - config.agents.keys()
    if missing:
        raise ValueError(f"models.yaml is missing agents: {sorted(missing)}")
    return config


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_model_config() -> ModelConfig:
    return load_model_config(get_settings().contentblitz_config)
