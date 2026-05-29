# FreeAlphaRadar — containerised Streamlit app.
# Zero secrets, zero API keys: nothing to configure at build or run time.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps needed by lxml / scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libxml2-dev libxslt1-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching. The slim runtime deps
# keep the image small; the app uses the lexicon sentiment backend.
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Optional: uncomment to bake in the real FinBERT + XGBoost stack (~3.9 GB).
# RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy the application.
COPY . .

# Seed the offline sample cache at build time so the image runs immediately,
# even with no outbound network access.
RUN python run_scorer.py --seed-sample --no-refresh --no-ml || true

EXPOSE 8501

# Basic container healthcheck against Streamlit's health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fs http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
