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
pip install -r requirements.txt          # slim runtime — uses lexicon sentiment
streamlit run streamlit_app.py
```

That's it. The app seeds a local sample cache on first launch, so the dashboard
is populated immediately — even with no internet. Click **Refresh Data &
Re-score** to pull live data from the free sources.

> **Optional — real FinBERT.** The slim install uses a deterministic lexicon
> sentiment backend (fast, dependency-light). To enable the pre-trained FinBERT
> model and the XGBoost skeleton, also run
> `pip install -r requirements-ml.txt` (~3.9 GB). The app behaves identically
> either way; only the sentiment engine swaps in.

> **Customising the screening universe.** The default tickers live in
> [`universe.txt`](universe.txt) (one or more per line, `#` comments allowed) —
> edit it to change the default screen with no code changes. It's a curated
> "under-the-radar deep tech" list, not a quote feed, so it doesn't need to
> auto-update; the *data* for those tickers is what refreshes (via the
> scheduler below). You can also just type any tickers into the sidebar at
> runtime to override it for a session.

### Make targets

```bash
make install     # install slim runtime dependencies
make install-ml  # also install FinBERT + XGBoost (optional, heavyweight)
make install-dev # runtime + test/format tooling
make run         # launch the Streamlit dashboard
make scorer      # run the batch scorer (warms the cache)
make seed        # (re)seed the offline sample dataset
make test        # run the offline test-suite (no network)
make format      # auto-format with black + isort
make lint        # check formatting
```

---

## 🧠 The scoring framework — "Moat + Momentum + Misvaluation"

35 factors across four groups, each **z-normalised within the current
universe** and combined via configurable sidebar sliders (equal-weight by
default).

| Group | Example factors |
|-------|-----------------|
| **Disruption & Moat** | Patent growth rate, R&D intensity, founder-led flag, product moat (manual), culture (manual) |
| **Growth & Momentum** | 3-yr revenue CAGR, **~5y & ~10y revenue CAGR + full-cycle margin trend (from full SEC XBRL history)**, gross-margin expansion, price momentum, employee growth, customer-concentration risk |
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
├── universe.txt                # Editable default screening universe
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

### Streamlit Community Cloud → `freealpharadar.streamlit.app`

There are **zero secrets**, and the only dependencies Streamlit Cloud installs
are the slim `requirements.txt` (the heavyweight ML stack stays in the optional
`requirements-ml.txt`), so the build is fast and reliable on the free tier.

1. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with
   GitHub → **Create app** → **Deploy a public app from GitHub**.
2. **Repository:** `chizkidd/freealpharadar` · **Branch:** `main` ·
   **Main file path:** `streamlit_app.py`.
3. In the **App URL** field, set the subdomain to `freealpharadar` so the public
   URL becomes **`https://freealpharadar.streamlit.app`**. (The subdomain must be
   globally unique; if it is taken, choose another. You can also change it later
   under **App settings → General → Custom subdomain**.)
4. Open **Advanced settings** and select **Python 3.11**. No secrets are needed.
5. Click **Deploy**. The app self-seeds sample data on first load, so the Radar
   Screen renders immediately while live data fetches in the background.

> Streamlit Cloud's filesystem is ephemeral: the SQLite cache and any watchlist
> changelogs reset when the container restarts. Sample data re-seeds
> automatically, so the app is always populated. For a persistent, pre-warmed
> cache use the scheduled refresh below.

You can also click the **Open in Streamlit** badge at the top of this README,
which pre-fills the repo, `main` branch and `streamlit_app.py`.

### Docker

```bash
docker compose up --build       # http://localhost:8501
```

### Scheduled refresh → pre-warmed cloud cache (GitHub Actions)

`.github/workflows/scheduler.yml` runs daily (and on-demand via the **Actions**
tab) to keep the hosted app showing **live data with no refresh wait**:

1. `run_scorer.py --no-ml --export-snapshot` fetches fresh data for the default
   universe and writes the warmed cache to a committed JSON snapshot,
   `data/prewarm_cache.json`.
2. The workflow commits that snapshot to the deploy branch, which triggers a
   Streamlit Cloud redeploy.
3. On boot the app calls `seed_from_snapshot()` (see `streamlit_app.py`) to load
   the snapshot into its SQLite cache, so the Radar Screen renders real,
   recent figures immediately. With no snapshot it falls back to the synthetic
   sample seed.

Why a JSON snapshot rather than the SQLite file? Streamlit Cloud's filesystem
is ephemeral (the DB resets on restart) and binary SQLite makes messy git
diffs, so the committed, diff-friendly JSON is the durable hand-off between the
scheduler and the app. No secrets are required for any of this. To pre-warm
locally on demand: `python run_scorer.py --export-snapshot`.

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
