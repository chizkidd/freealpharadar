# 📡 FreeAlphaRadar

> Zero-cost alpha discovery engine. Systematically surface the **next Palantir,
> SanDisk, or Bloom Energy** using **only free, public data** — no API keys, no
> sign-ups, no payment, ever.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=chizkidd/freealpharadar&branch=main&mainModule=streamlit_app.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/freealpharadar/blob/main/colab_setup.ipynb)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

FreeAlphaRadar ingests financial, alternative and qualitative data from
entirely zero-cost public feeds and produces a **ranked, transparently scored**
list of high-potential, under-the-radar companies — ready for an investment
committee discussion. Every score is fully drillable: raw value → normalised
z-score → weighted contribution.

---

## ✨ Why it's different

| Principle | What it means here |
|-----------|--------------------|
| **100% free data** | yfinance, SEC EDGAR, PatentsView, GDELT — all key-less and free. |
| **Zero configuration** | No `.env`, no secrets, no registration. `streamlit run` and go. |
| **Works fully offline** | First launch seeds a SQLite cache with sample data; every fetcher falls back to cache on failure. |
| **Totally transparent** | A waterfall chart decomposes every company's score, factor by factor. |
| **Zero-shot ML** | Pre-trained FinBERT + PCA/K-Means clustering. Graceful lexicon fallback when offline. |

---

## 🚀 Quick start

```bash
git clone https://github.com/chizkidd/freealpharadar.git
cd freealpharadar
pip install -r requirements.txt
streamlit run streamlit_app.py
```

That's it. The app seeds a local sample cache on first launch, so the dashboard
is populated immediately — even with no internet. Click **Refresh Data &
Re-score** to pull live data from the free sources.

### Make targets

```bash
make install   # install pinned dependencies
make run        # launch the Streamlit dashboard
make scorer     # run the batch scorer (warms the cache)
make seed       # (re)seed the offline sample dataset
make test       # run the offline test-suite (no network)
make format     # auto-format with black + isort
make lint       # check formatting
```

---

## 🧠 The scoring framework — "Moat + Momentum + Misvaluation"

32 factors across four groups, each **z-normalised within the current
universe** and combined via configurable sidebar sliders (equal-weight by
default).

| Group | Example factors |
|-------|-----------------|
| **Disruption & Moat** | Patent growth rate, R&D intensity, founder-led flag, product moat (manual), culture (manual) |
| **Growth & Momentum** | 3-yr revenue CAGR, gross-margin expansion, price momentum, employee growth, customer-concentration risk |
| **Valuation & Inefficiency** | EV/Gross-Profit, P/E, P/B, short interest, institutional ownership, insider activity, Piotroski F-score |
| **Qualitative Flags** | GDELT news tone, FinBERT controversy score, key-person dependency, regulatory-risk count |

> **Under-the-radar lens:** factors like *institutional ownership* are oriented
> so that **lower** is more attractive — the hallmark of a genuinely
> overlooked name with room to re-rate.

---

## 📊 The dashboard

* **🛰️ Radar Screen** — Plotly scatter of Score vs. Market Cap, coloured by
  sector, with hover company cards and live filters for sector, market-cap
  range and score threshold.
* **🔬 Deep Dive** — financial trajectory, SEC risk-factor excerpt with FinBERT
  sentiment highlighting, patent timeline + top assignees, GDELT news feed with
  tone bars, and a **waterfall** of the score breakdown.
* **⭐ Watchlist** — save/remove companies (SQLite), then **Check for Changes**
  to re-score and write a changelog to `watchlist_changes/<ticker>_<date>.txt`,
  with deltas shown inline.

---

## 🗂️ Data sources (all free, no keys)

| Source | Library / endpoint | Used for |
|--------|--------------------|----------|
| **yfinance** | `yfinance` | Prices, income/balance/cash-flow statements, short interest, ownership |
| **SEC EDGAR** | `data.sec.gov` + `sec-edgar-downloader` | Risk factors, MD&A, business description, Form 4 insider data, XBRL facts |
| **PatentsView** | `search.patentsview.org` (45 req/min, no key) | Patent counts, assignees, titles over time |
| **GDELT 2.0** | `api.gdeltproject.org` Doc API | News sentiment (tone) and volume |
| **Manual CSV** | optional upload | Employee/culture/product-moat signals — gracefully ignored if absent |

Everything is cached in SQLite with per-source TTLs (configurable via env vars)
so free-tier rate limits are respected and the app remains usable offline.

---

## 🏗️ Project structure

```
freealpharadar/
├── streamlit_app.py            # Streamlit entrypoint (stateless, cached)
├── run_scorer.py               # Batch scorer for cron / GitHub Actions
├── manual_upload_template.csv  # Optional manual-signals template
├── colab_setup.ipynb           # One-click Colab/Kaggle launcher
├── Dockerfile / docker-compose.yml
├── requirements.txt            # Pinned for reproducibility
├── freealpharadar/
│   ├── config.py               # Zero-config settings (no secrets)
│   ├── database.py             # SQLite cache + watchlist + scores
│   ├── pipeline.py             # Per-company data orchestration
│   ├── service.py              # ingest → enrich → score pipeline
│   ├── sample_data.py          # Deterministic offline sample data
│   ├── watchlist.py            # Re-scoring + changelog writing
│   ├── fetchers/               # yfinance, SEC, PatentsView, GDELT, manual CSV
│   ├── scoring/                # factors, normalisation, engine
│   ├── ml/                     # FinBERT, clustering, XGBoost skeleton, enrich
│   ├── ui/                     # radar screen, deep dive, watchlist, sidebar
│   └── utils/                  # logging
├── tests/                      # offline pytest suite (no network)
└── data/sample/                # canned sample dataset
```

---

## 🤖 AI / ML

* **FinBERT** (`ProsusAI/finbert`) classifies SEC risk-section and GDELT
  headline sentiment into a per-company controversy score. If the model can't
  be downloaded (offline / constrained runtime), a deterministic finance
  lexicon backend takes over with an identical API.
* **PCA + K-Means** clusters companies on financial ratios so you can spot names
  that behave unlike their nominal peers.
* **XGBoost** breakout model is a thin, optional skeleton: supply a
  `ticker,label` CSV of historical breakouts to train a re-ranker; with no
  labels (the default) the system uses rule-based scoring with **zero**
  degradation.

---

## ☁️ Deployment

### Streamlit Community Cloud (one click, no secrets)

Click the **Open in Streamlit** badge above, or point Streamlit Cloud at this
repo with main module `streamlit_app.py`. Because there are **zero secrets**,
it works out of the box.

### Docker

```bash
docker compose up --build       # http://localhost:8501
```

### Scheduled refresh (GitHub Actions)

`.github/workflows/scheduler.yml` runs `run_scorer.py` daily, warms the SQLite
cache, and commits it back so a hosted app reads pre-computed data. No secrets
required — trigger it manually any time from the **Actions** tab.

---

## 🧪 Testing & quality

```bash
make test          # FAR_OFFLINE=1 pytest — no network, fully reproducible
make lint          # black --check + isort --check
pre-commit install # black, isort, hygiene hooks on every commit
```

The suite (`tests/test_scoring.py`) verifies scoring maths, factor parsing,
normalisation, FinBERT fallback and the end-to-end offline pipeline using
canned data only.

---

## ⚙️ Configuration (all optional)

There is **nothing you must configure.** A few knobs are exposed as environment
variables for convenience:

| Variable | Default | Purpose |
|----------|---------|---------|
| `FAR_OFFLINE` | `false` | Force cache/sample-only mode (no network). |
| `FAR_DB_PATH` | `data/freealpharadar.sqlite` | SQLite cache location. |
| `FAR_TTL_PRICES` / `_FUNDAMENTALS` / `_SEC` / `_PATENTS` / `_NEWS` | varies | Per-source cache TTLs (seconds). |
| `FAR_LOG_LEVEL` | `INFO` | Logging verbosity. |

---

## ⚠️ Disclaimer

FreeAlphaRadar is a research and educational tool. The bundled sample data is
**synthetic** and the live data is provided "as is" from third-party public
sources. **Nothing here is investment advice.** Do your own due diligence.

## License

MIT — see [LICENSE](LICENSE).
