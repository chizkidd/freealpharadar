# 📡 FreeAlphaRadar

[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit&logoColor=white)](https://freealpharadar.streamlit.app/)
[![Daily Data Refresh & Re-score](https://github.com/chizkidd/freealpharadar/actions/workflows/scheduler.yml/badge.svg)](https://github.com/chizkidd/freealpharadar/actions/workflows/scheduler.yml)
[![Weekly Discovery](https://github.com/chizkidd/freealpharadar/actions/workflows/discover.yml/badge.svg)](https://github.com/chizkidd/freealpharadar/actions/workflows/discover.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/freealpharadar/blob/main/colab_setup.ipynb)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=chizkidd.freealpharadar)

FreeAlphaRadar ingests financial, alternative and qualitative data from
entirely zero-cost public feeds and produces a **ranked, transparently scored**
list of high-potential, under-the-radar companies — ready for an investment
committee discussion. Every score is fully drillable: raw value → normalised
z-score → weighted contribution. 

>This project is for **educational purposes only** and is not intended for real trading or investment.

---

## ✨ Why it's different

| Principle | What it means here |
|-----------|--------------------|
| **100% free data** | yfinance, SEC EDGAR, GDELT — key-less and free (PatentsView needs a free, optional key). |
| **Zero configuration** | No `.env`, no secrets, no registration. `streamlit run` and go. |
| **Works fully offline** | First launch seeds a SQLite cache with sample data; every fetcher falls back to cache on failure. |
| **Totally transparent** | A waterfall chart decomposes every company's score, factor by factor. |
| **Zero-shot ML** | Pre-trained FinBERT + PCA/K-Means clustering. Graceful lexicon fallback when offline. |
| **Self-discovering** | An optional offline job scans **all ~8,000 SEC filers** and regenerates the watchlist with the ranked top under-the-radar names. |

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
> edit it to change the default screen with no code changes. By default it's a
> hand-curated "under-the-radar deep tech" list; you can also type any tickers
> into the sidebar to override it for a session, or opt into the
> [auto-discovery job](#-auto-discovered-universe-optional-offline) below, which
> regenerates `universe.txt` from a market-wide scan.

### Make targets

```bash
make install            # slim runtime deps (lexicon sentiment)
make install-ml         # + FinBERT + XGBoost (optional, heavyweight)
make install-dev        # runtime + pytest/black/isort
make install-warehouse  # + duckdb/pyarrow for the discovery job
make run                # launch the Streamlit dashboard
make scorer             # batch scorer (warms the cache)
make seed               # (re)seed the offline sample dataset
make warehouse          # build the bulk SEC fundamentals store (needs network)
make discover           # scan all filers → promote ranked top-10 (needs network)
make test               # offline test-suite (no network)
make lint / make format # black + isort (check / apply)
make docker-up          # run via docker compose
make clean              # remove caches/build artefacts
```

### Dependency layers

Dependencies are split so the cloud install stays lean; each layer is additive.

| File | Installs | When |
|------|----------|------|
| `requirements.txt` | streamlit, pandas, plotly, yfinance, scikit-learn… | always (the only file Streamlit Cloud installs) |
| `requirements-ml.txt` | transformers, torch, xgboost (~3.9 GB) | optional — real FinBERT + XGBoost locally |
| `requirements-dev.txt` | pytest, black, isort, pre-commit | tests / contributing |
| `requirements-warehouse.txt` | duckdb, pyarrow | the optional market-wide discovery job |

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
| **SEC EDGAR** | `data.sec.gov` JSON + EDGAR archives (no key) | Risk factors, MD&A, business description, Form 4 insider data, XBRL facts |
| **PatentsView** | `search.patentsview.org` (free API key, optional) | Patent counts, assignees, titles over time |
| **GDELT 2.0** | `api.gdeltproject.org` Doc API | News sentiment (tone) and volume |
| **Manual CSV** | optional upload | Employee/culture/product-moat signals — gracefully ignored if absent |

Everything is cached in SQLite with per-source TTLs (configurable via env vars)
so free-tier rate limits are respected and the app remains usable offline. For
resilience from datacenter/CI IPs, yfinance uses a browser-impersonating
`curl_cffi` session and falls back to **Stooq** for prices (with a SEC
shares × price market-cap approximation) when Yahoo returns nothing.

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
│   ├── warehouse/              # optional bulk SEC fundamentals (DuckDB/Parquet)
│   ├── discovery/              # optional market-wide screen → ranked top-N
│   └── utils/                  # logging
├── discoveries/                # dated auto-discovery reports
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

## 🛰️ Auto-discovered universe (optional, offline)

Beyond the curated list, FreeAlphaRadar can **scan the entire market and pick
its own watchlist** — "from all ~8,000 SEC filers, the top-10 under-the-radar
names, ranked best→worst." It's a two-stage **offline** funnel (never runs
inside the live app) built on free SEC bulk data:

1. **Warehouse** (`freealpharadar/warehouse/`) — downloads SEC's free
   **Financial Statement Data Sets** (quarterly, 2009→present, every XBRL
   filer incl. delisted) into a gitignored DuckDB/Parquet store.
2. **Stage 1 — bulk screen** (`discovery/screen.py`) — one DuckDB pass over all
   filers computing cheap fundamentals (revenue CAGR, margin trend, R&D
   intensity) with under-the-radar gates (small/mid revenue scale + sustained
   growth) → ~100-name shortlist.
3. **Stage 2 — full scoring** (`discovery/discover.py`) — runs the **existing
   35-factor pipeline** on the shortlist, applies a market-cap ceiling, and
   ranks → **top-10**.
4. **Promote** — rewrites `universe.txt`, regenerates the prewarm snapshot, and
   writes `discoveries/<date>.md`. The live app then shows the self-discovered
   names.

```bash
pip install -r requirements.txt -r requirements-warehouse.txt   # duckdb, pyarrow
make warehouse                       # build the store (needs network)
python -m freealpharadar.discovery run --top 10
```

It runs weekly via `.github/workflows/discover.yml` (or on-demand), committing
the refreshed `universe.txt` + snapshot + report — so the deployed app's list
updates itself. Heavyweight by design (multi-GB store, needs network), opt-in,
and entirely outside the app's hot path.

---

## 🧪 Testing & quality

```bash
make test          # FAR_OFFLINE=1 pytest — no network, fully reproducible
make lint          # black --check + isort --check
pre-commit install # black, isort, hygiene hooks on every commit
```

~50+ tests run fully offline (`FAR_OFFLINE=1`, no network):

* `tests/test_scoring.py` — normalisation, all 35 factors (incl. the SEC
  long-horizon ones), the engine, the FinBERT lexicon fallback, the
  universe loader, and the end-to-end offline pipeline on canned data.
* `tests/test_warehouse.py` — the bulk loader, DuckDB store, Stage-1 screen
  gates and discovery promotion against a synthetic SEC ZIP fixture (auto-
  skipped if `duckdb`/`pyarrow` aren't installed).

---

## ⚙️ Configuration (all optional)

There is **nothing you must configure.** A few knobs are exposed as environment
variables for convenience:

| Variable | Default | Purpose |
|----------|---------|---------|
| `FAR_OFFLINE` | `false` | Force cache/sample-only mode (no network). |
| `FAR_DB_PATH` | `data/freealpharadar.sqlite` | SQLite cache location. |
| `FAR_TTL_FUNDAMENTALS` / `_SEC` / `_PATENTS` / `_NEWS` | 1d / 1w / 1w / 12h | Per-source cache TTLs (seconds). |
| `FAR_CONCURRENCY` / `FAR_MAX_RETRIES` / `FAR_HTTP_TIMEOUT` | `5` / `4` / `30` | Async fetch concurrency, retries, per-request timeout. |
| `FAR_FINBERT_MODEL` | `ProsusAI/finbert` | HuggingFace sentiment model id. |
| `FAR_SEC_USER_AGENT` | research UA | Identifies you to SEC EDGAR (set a real contact to reduce throttling). |
| `FAR_PATENTSVIEW_API_KEY` | _(unset)_ | Optional [free PatentsView key](https://patentsview.org/apis/keyrequest) — enables the Patents tab; omit to skip patents. |
| `FAR_GDELT_INTERVAL` | `2.0` | Seconds between GDELT calls (raise if you still hit `429`s). |
| `FAR_LOG_LEVEL` | `INFO` | Logging verbosity. |

---

## ⚠️ Disclaimer

FreeAlphaRadar is a research and educational tool. The bundled sample data is
**synthetic** and the live data is provided "as is" from third-party public
sources. **Nothing here is investment advice.** Do your own due diligence.

## License

MIT — see [LICENSE](LICENSE).
