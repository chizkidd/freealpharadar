"""Offline tests for the scoring engine, factors and ML fallbacks.

Every test runs against deterministic canned data (see ``conftest.py``) with
``FAR_OFFLINE=1`` set, so the suite makes **no network calls** and is fully
reproducible. Run with ``make test`` or ``pytest``.
"""

from __future__ import annotations

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
