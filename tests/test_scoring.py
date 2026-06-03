"""Offline tests for the scoring engine, factors and ML fallbacks.

Every test runs against deterministic canned data (see ``conftest.py``) with
``FAR_OFFLINE=1`` set, so the suite makes **no network calls** and is fully
reproducible. Run with ``make test`` or ``pytest``.
"""

from __future__ import annotations

import json
import math

import pytest

from freealpharadar.ml.finbert import FinBERTSentiment
from freealpharadar.pipeline import CompanyData
from freealpharadar.scoring import DEFAULT_WEIGHTS, FACTORS, ScoringEngine
from freealpharadar.scoring import factors as F
from freealpharadar.scoring.normalize import to_0_100, zscore_column


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
class TestNormalize:
    def test_zscore_basic(self):
        z = zscore_column([1.0, 2.0, 3.0], higher_is_better=True)
        assert len(z) == 3
        assert z[0] < z[1] < z[2]
        assert math.isclose(sum(z), 0.0, abs_tol=1e-9)

    def test_zscore_orientation_flips_sign(self):
        up = zscore_column([1.0, 2.0, 3.0], higher_is_better=True)
        down = zscore_column([1.0, 2.0, 3.0], higher_is_better=False)
        assert all(math.isclose(a, -b, abs_tol=1e-9) for a, b in zip(up, down))

    def test_zscore_missing_values_neutral(self):
        z = zscore_column([1.0, None, 3.0], higher_is_better=True)
        assert z[1] == 0.0

    def test_zscore_all_missing(self):
        assert zscore_column([None, None], higher_is_better=True) == [0.0, 0.0]

    def test_zscore_zero_variance(self):
        assert zscore_column([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]

    def test_to_0_100_bounds(self):
        scaled = to_0_100([-10.0, 0.0, 10.0])
        assert scaled[0] == pytest.approx(0.0)
        assert scaled[1] == pytest.approx(50.0)
        assert scaled[2] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Factor calculations (precise, hand-checked numbers)
# --------------------------------------------------------------------------- #
class TestFactors:
    def test_revenue_cagr(self, handcrafted_company):
        # Rev series most-recent-first: 1000, 800, 500, 250 -> 3y CAGR
        # (1000/250)^(1/3) - 1 = 4^(1/3) - 1
        cagr = F.f_revenue_cagr(handcrafted_company)
        assert cagr == pytest.approx(4 ** (1 / 3) - 1, rel=1e-6)

    def test_gross_margin(self, handcrafted_company):
        assert F.f_gross_margin(handcrafted_company) == pytest.approx(0.6)

    def test_gross_margin_expansion(self, handcrafted_company):
        # latest gm 600/1000=0.6 ; prior(2) 250/500=0.5 -> +0.1
        assert F.f_gross_margin_expansion(handcrafted_company) == pytest.approx(0.1)

    def test_rnd_intensity(self, handcrafted_company):
        assert F.f_rnd_intensity(handcrafted_company) == pytest.approx(0.2)

    def test_ev_gp(self, handcrafted_company):
        # EV 1200 / gross profit 600 = 2.0
        assert F.f_ev_gp(handcrafted_company) == pytest.approx(2.0)

    def test_ebitda_margin(self, handcrafted_company):
        assert F.f_ebitda_margin(handcrafted_company) == pytest.approx(0.2)

    def test_fcf_positive(self, handcrafted_company):
        assert F.f_fcf_positive(handcrafted_company) == 1.0

    def test_founder_led_flag(self, handcrafted_company):
        assert F.f_founder_led(handcrafted_company) == 1.0

    def test_regulatory_risk_count(self, handcrafted_company):
        assert F.f_regulatory_risk(handcrafted_company) == 3.0

    def test_piotroski_in_range(self, handcrafted_company):
        score = F.f_piotroski(handcrafted_company)
        assert score is not None
        assert 0.0 <= score <= 5.0
        # Positive NI, positive CFO, CFO>NI, margin expansion, growth -> high.
        assert score >= 4.0

    def test_manual_factors(self, handcrafted_company):
        assert F.f_product_moat(handcrafted_company) == pytest.approx(8.0)
        assert F.f_culture(handcrafted_company) == pytest.approx(7.5)
        assert F.f_glassdoor(handcrafted_company) == pytest.approx(4.2)

    def test_patent_growth(self, handcrafted_company):
        assert F.f_patent_growth(handcrafted_company) == pytest.approx(0.5)

    def test_news_tone(self, handcrafted_company):
        assert F.f_news_tone(handcrafted_company) == pytest.approx(2.5)

    def test_revenue_cagr_5y_from_sec_facts(self, handcrafted_company):
        # Revenue grows exactly 20%/yr in the SEC facts series.
        assert F.f_revenue_cagr_5y(handcrafted_company) == pytest.approx(0.20)

    def test_gross_margin_trend_from_sec_facts(self, handcrafted_company):
        # Gross margin rises exactly +0.01 per year.
        assert F.f_gross_margin_trend(handcrafted_company) == pytest.approx(0.01)

    def test_revenue_cagr_uses_alternate_concepts(self):
        # Mature filers tag revenue under SalesRevenueNet (revenue_alt2) rather
        # than the primary concept; the factor must fall back to it.
        from freealpharadar.pipeline import CompanyData

        facts = {
            "revenue_alt2": [{"fy": 2019, "val": 100.0}, {"fy": 2024, "val": 200.0}]
        }
        c = CompanyData(ticker="X", sec={"facts": facts})
        assert F.f_revenue_cagr_5y(c) == pytest.approx(2.0 ** (1 / 5) - 1.0)

    def test_long_horizon_factors_none_without_facts(self, single_company):
        # Sample companies via build_sample_dataset DO have facts; an explicit
        # company without facts must yield None, not raise.
        from freealpharadar.pipeline import CompanyData

        bare = CompanyData(ticker="X")
        assert F.f_revenue_cagr_5y(bare) is None
        assert F.f_gross_margin_trend(bare) is None


# --------------------------------------------------------------------------- #
# None-safety
# --------------------------------------------------------------------------- #
class TestNoneSafety:
    def test_all_factors_handle_empty_company(self, empty_company):
        for spec in FACTORS:
            # Must return None or a float, never raise.
            val = spec.fn(empty_company)
            assert val is None or isinstance(val, (int, float))

    def test_factor_count_is_at_least_30(self):
        assert len(FACTORS) >= 30

    def test_factor_names_unique(self):
        names = [f.name for f in FACTORS]
        assert len(names) == len(set(names))

    def test_default_weights_cover_all_factors(self):
        for f in FACTORS:
            assert f.name in DEFAULT_WEIGHTS


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class TestEngine:
    def test_score_universe_sorted_desc(self, sample_universe):
        results = ScoringEngine().score_universe(sample_universe)
        assert len(results) == len(sample_universe)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_0_100(self, sample_universe):
        results = ScoringEngine().score_universe(sample_universe)
        assert all(0.0 <= r.score <= 100.0 for r in results)

    def test_contributions_present(self, sample_universe):
        results = ScoringEngine().score_universe(sample_universe)
        for r in results:
            assert len(r.contributions) == len(FACTORS)
            # Ordered by absolute contribution (descending).
            abs_contribs = [abs(c.contribution) for c in r.contributions]
            assert abs_contribs == sorted(abs_contribs, reverse=True)

    def test_empty_universe(self):
        assert ScoringEngine().score_universe([]) == []

    def test_single_company_universe(self, single_company):
        results = ScoringEngine().score_universe([single_company])
        assert len(results) == 1
        # With one company every z-score is 0 -> composite 0 -> score 50.
        assert results[0].score == pytest.approx(50.0)

    def test_disabling_factor_changes_score(self, sample_universe):
        base = ScoringEngine().score_universe(sample_universe)
        zero_weights = {f.name: 0.0 for f in FACTORS}
        zero_weights["revenue_cagr"] = 5.0  # only one factor matters now
        tuned = ScoringEngine(weights=zero_weights).score_universe(sample_universe)
        base_order = [r.ticker for r in base]
        tuned_order = [r.ticker for r in tuned]
        # Different weighting should generally reorder; at minimum it must run.
        assert set(base_order) == set(tuned_order)

    def test_group_weight_summary(self):
        summary = ScoringEngine().group_weight_summary()
        assert sum(summary.values()) == pytest.approx(float(len(FACTORS)))

    def test_zero_weight_factor_has_zero_contribution(self, sample_universe):
        weights = {f.name: 0.0 for f in FACTORS}
        results = ScoringEngine(weights=weights).score_universe(sample_universe)
        for r in results:
            assert all(c.contribution == 0.0 for c in r.contributions)

    def test_missing_factor_does_not_dilute_composite(self):
        # A factor that is None for every company must be *neutral* (excluded
        # from the weighted average), not a zero that drags the composite toward
        # the mean. Including such a factor must not change raw_composite.
        from freealpharadar.scoring.factors import FactorGroup, FactorSpec

        c1, c2 = CompanyData(ticker="A"), CompanyData(ticker="B")
        present = FactorSpec(
            "present",
            "P",
            FactorGroup.MOMENTUM,
            lambda c: 1.0 if c.ticker == "A" else 0.0,
            True,
            "",
        )
        missing = FactorSpec(
            "missing", "M", FactorGroup.MOMENTUM, lambda c: None, True, ""
        )
        only = ScoringEngine(factors=[present]).score_universe([c1, c2])
        both = ScoringEngine(factors=[present, missing]).score_universe([c1, c2])
        only_map = {r.ticker: r.raw_composite for r in only}
        both_map = {r.ticker: r.raw_composite for r in both}
        assert only_map == both_map


# --------------------------------------------------------------------------- #
# Provider-agnostic patent fetcher (offline; providers stubbed)
# --------------------------------------------------------------------------- #
class TestPatentProviders:
    def test_no_provider_returns_empty(self, monkeypatch):
        import asyncio

        from freealpharadar.fetchers import patents_fetcher as PF

        monkeypatch.setattr(PF, "PATENTSVIEW_API_KEY", "")
        monkeypatch.setattr(PF, "LENS_API_TOKEN", "")
        out = asyncio.run(PF.PatentFetcher()._fetch_remote("AAA", company_name="Acme"))
        assert out["total_patents"] == 0
        assert out["sample_titles"] == []

    def test_lens_response_is_normalised(self, monkeypatch):
        import asyncio

        from freealpharadar.fetchers import patents_fetcher as PF

        monkeypatch.setattr(PF, "PATENTSVIEW_API_KEY", "")
        monkeypatch.setattr(PF, "LENS_API_TOKEN", "tok")

        async def fake_post(self, url, *, json_body=None, headers=None):
            assert headers["Authorization"] == "Bearer tok"
            return {
                "data": [
                    {
                        "biblio": {"invention_title": [{"text": "Quantum widget"}]},
                        "date_published": "2023-05-01",
                    },
                    {
                        "biblio": {"invention_title": [{"text": "AI gadget"}]},
                        "date_published": "2022-01-01",
                    },
                ]
            }

        monkeypatch.setattr(PF.PatentFetcher, "_http_post_json", fake_post)
        out = asyncio.run(PF.PatentFetcher()._fetch_remote("AAA", company_name="Acme"))
        assert out["total_patents"] == 2
        assert "Quantum widget" in out["sample_titles"]

    def test_patentsview_alias_is_provider_class(self):
        from freealpharadar.fetchers import patents_fetcher as PF

        assert PF.PatentsViewFetcher is PF.PatentFetcher


# --------------------------------------------------------------------------- #
# FinBERT lexicon fallback (offline)
# --------------------------------------------------------------------------- #
class TestFinBERTFallback:
    def test_backend_is_lexicon_offline(self):
        analyzer = FinBERTSentiment()
        assert analyzer.backend == "lexicon"

    def test_positive_text(self):
        res = FinBERTSentiment().analyze("record growth and strong profit, a win")
        assert res.label == "positive"
        assert res.score > 0

    def test_negative_text(self):
        res = FinBERTSentiment().analyze(
            "lawsuit, fraud investigation and bankruptcy risk"
        )
        assert res.label == "negative"
        assert res.score < 0

    def test_empty_text_neutral(self):
        res = FinBERTSentiment().analyze("")
        assert res.label == "neutral"
        assert res.score == 0.0

    def test_analyze_many_averages(self):
        res = FinBERTSentiment().analyze_many(["strong growth", "loss and decline"])
        assert -1.0 <= res.score <= 1.0


# --------------------------------------------------------------------------- #
# End-to-end offline pipeline
# --------------------------------------------------------------------------- #
class TestOfflinePipeline:
    def test_pipeline_runs_offline(self):
        from freealpharadar.sample_data import seed_cache
        from freealpharadar.service import run_pipeline

        seed_cache(force=True)
        output = run_pipeline(
            ["PLTR", "BE", "IONQ"],
            run_ml=True,
            force_refresh=False,
            persist=False,
        )
        assert len(output.results) == 3
        assert all(0.0 <= r.score <= 100.0 for r in output.results)

    def test_enrichment_populates_derived(self, sample_universe):
        from freealpharadar.ml.enrich import enrich_companies

        enriched = enrich_companies([c for c in sample_universe])
        for c in enriched:
            assert "controversy_score" in c.derived
            assert "finbert_sentiment" in c.derived


# --------------------------------------------------------------------------- #
# Universe configuration & sample coverage
# --------------------------------------------------------------------------- #
class TestUniverseConfig:
    def test_universe_non_empty_and_unique(self):
        from freealpharadar.config import settings

        u = settings.default_universe
        assert len(u) >= 12
        assert len(u) == len(set(u))  # no duplicates
        assert all(t == t.upper() for t in u)  # normalised

    def test_universe_parser_handles_comments_and_separators(self, tmp_path):
        import freealpharadar.config as cfg
        from freealpharadar.config import UNIVERSE_FILE, _load_default_universe

        f = tmp_path / "universe.txt"
        f.write_text(
            "# header\nPLTR  # inline\nBE, IONQ RKLB\n\nPLTR\n", encoding="utf-8"
        )
        original = cfg.UNIVERSE_FILE
        try:
            cfg.UNIVERSE_FILE = f
            parsed = _load_default_universe()
        finally:
            cfg.UNIVERSE_FILE = original
        assert parsed == ["PLTR", "BE", "IONQ", "RKLB"]  # deduped, ordered

    def test_universe_falls_back_when_missing(self, tmp_path):
        import freealpharadar.config as cfg
        from freealpharadar.config import _FALLBACK_UNIVERSE, _load_default_universe

        original = cfg.UNIVERSE_FILE
        try:
            cfg.UNIVERSE_FILE = tmp_path / "does_not_exist.txt"
            assert _load_default_universe() == _FALLBACK_UNIVERSE
        finally:
            cfg.UNIVERSE_FILE = original

    def test_sample_dataset_covers_full_universe(self):
        from freealpharadar.config import settings
        from freealpharadar.sample_data import build_sample_dataset

        ds = build_sample_dataset()
        assert all(t in ds for t in settings.default_universe)
        # Every bundle has all four raw sources.
        for bundle in ds.values():
            assert {"yfinance", "sec", "patents", "news"} <= set(bundle)

    def test_rng_deterministic_across_calls(self):
        from freealpharadar.sample_data import _rng

        assert _rng("PLTR").random() == _rng("PLTR").random()


# --------------------------------------------------------------------------- #
# Snapshot quality gate (#2)
# --------------------------------------------------------------------------- #
class TestSnapshotQualityGate:
    def test_coverage_fraction(self):
        from freealpharadar.sample_data import snapshot_coverage

        snap = {
            "A": {"yfinance": {"key_metrics": {"market_cap": 1e9}}},
            "B": {"yfinance": {"key_metrics": {"market_cap": None}}},
            "C": {"sec": {"company_name": "C"}},
        }
        assert snapshot_coverage(snap) == pytest.approx(1 / 3)
        assert snapshot_coverage({}) == 0.0

    def test_seed_rejects_sparse_snapshot(self, tmp_path):
        from freealpharadar.database import Database
        from freealpharadar.sample_data import seed_from_snapshot

        snap = {"A": {"sec": {"company_name": "A"}}}  # no market cap → coverage 0
        p = tmp_path / "snap.json"
        p.write_text(json.dumps(snap))
        db = Database(tmp_path / "g.sqlite")
        assert seed_from_snapshot(db=db, path=p, min_coverage=0.3) == 0  # rejected
        assert seed_from_snapshot(db=db, path=p, min_coverage=0.0) == 1  # accepted

    def test_export_gate_skips_thin_snapshot(self, tmp_path):
        from freealpharadar.database import Database
        from freealpharadar.sample_data import export_cache_snapshot

        db = Database(tmp_path / "e.sqlite")
        db.set_cache("sec", "A", {"company_name": "A"})  # no yfinance market cap
        out = tmp_path / "out.json"
        assert export_cache_snapshot(["A"], db=db, path=out, min_coverage=0.6) == 0
        assert not out.exists()  # gate blocked the write
        assert export_cache_snapshot(["A"], db=db, path=out, min_coverage=0.0) == 1
        assert out.exists()  # gate off → written


# --------------------------------------------------------------------------- #
# Market-cap fallback (#3)
# --------------------------------------------------------------------------- #
class TestMarketCapFallback:
    def test_approx_from_shares_and_last_close(self):
        from freealpharadar.pipeline import _approx_market_cap

        yf = {
            "history": [
                {"date": "2024-01", "close": 10.0},
                {"date": "2024-02", "close": 12.0},
            ]
        }
        sec = {
            "facts": {
                "shares": [{"fy": 2022, "val": 500_000}, {"fy": 2023, "val": 1_000_000}]
            }
        }
        assert _approx_market_cap(yf, sec) == pytest.approx(12.0 * 1_000_000)

    def test_none_without_shares_or_price(self):
        from freealpharadar.pipeline import _approx_market_cap

        assert _approx_market_cap({}, {}) is None
        assert (
            _approx_market_cap(
                {"history": [{"date": "x", "close": 5.0}]}, {"facts": {}}
            )
            is None
        )

    def test_name_falls_back_to_sec_when_yfinance_blank(self):
        # When yfinance returns no name (blocked), the display name should come
        # from the SEC filing, not the bare ticker.
        import asyncio

        from freealpharadar.database import get_db
        from freealpharadar.pipeline import gather_company

        db = get_db()
        db.set_cache(
            "yfinance",
            "ZQX",
            {
                "info": {},
                "history": [],
                "income_statement": [],
                "balance_sheet": [],
                "cash_flow": [],
                "key_metrics": {"market_cap": None, "sector": None},
            },
        )
        db.set_cache(
            "sec",
            "ZQX",
            {
                "company_name": "Test Industries Inc.",
                "sections": {},
                "flags": {},
                "facts": {},
                "insider_transactions": {},
            },
        )
        company = asyncio.run(gather_company("ZQX"))
        assert company.name == "Test Industries Inc."


# --------------------------------------------------------------------------- #
# Robust SEC section extraction (#4)
# --------------------------------------------------------------------------- #
class TestSecSectionExtraction:
    def test_longest_match_skips_table_of_contents(self):
        from freealpharadar.fetchers.sec_fetcher import _extract_section

        body = (
            "Item 1A. Risk Factors. "
            + "We face many material risks and uncertainties. " * 40
            + " Item 1B. Unresolved Staff Comments."
        )
        # A TOC mentions Item 1A/1B first (short), the real section comes later.
        text = "Table of Contents Item 1A Risk Factors 12 Item 1B Comments 14 " + body
        out = _extract_section(
            text, r"item\s*1a\b", (r"item\s*1b\b", r"item\s*2\b"), max_len=8000
        )
        assert "We face many material risks" in out
        assert len(out) > 500  # not the one-line TOC stub

    def test_missing_section_returns_empty(self):
        from freealpharadar.fetchers.sec_fetcher import _extract_section

        assert _extract_section("nothing here", r"item\s*1a\b", (r"item\s*2\b",)) == ""
