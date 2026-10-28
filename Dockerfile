FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    CONTENTBLITZ_CONFIG=/app/config/models.yaml \
    CONTENTBLITZ_DATA_DIR=/data \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# textstat needs nltk's cmudict for syllable counts and downloads it on first use if missing.
# Fetch it now, so the container never downloads at runtime.
RUN python -m nltk.downloader -d "$NLTK_DATA" cmudict \
    && python -c "import nltk; nltk.data.find('corpora/cmudict')"

COPY config ./config
COPY src ./src

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data \
    && chown app:app /data
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/contentblitz/web_app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
